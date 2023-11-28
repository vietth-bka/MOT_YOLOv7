import numpy as np
from collections import deque, defaultdict
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

# from .kalman_filter import KalmanFilter
from tracker import matching
from .basetrack import BaseTrack, TrackState, MCBaseTrack
from utils.general import scale_coords
from .custom_kalman_filter import *
from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


class MCTrack(MCBaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls_id, feat=None, feat_history=50):
        """
        :param tlwh:
        :param score:
        """
        # object class id
        self.cls_id = cls_id

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        # ----- init is_activated to be False
        self.is_activated = False

        self.score = score
        self.track_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9
        if feat is not None:
            # self.update_features(feat)
            self.update_features_custom(feat)
        self.features = deque([], maxlen=feat_history)

        self.age = 0
        self.time_since_last_update = 0  # unit: number of frames

        self.hit_streak = 0

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_features_custom(self, feat, score=None):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        new_alpha = self.alpha
        if score is not None:
            new_alpha = -0.5 * score + 1.2
            if new_alpha > 0.95:
                new_alpha = 0.95
            if new_alpha < 0.9:
                new_alpha = 0.9

        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = new_alpha * self.smooth_feat + (1 - new_alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def reset_track_id(self):
        """
        :return:
        """
        self.reset_track_id(self.cls_id)

    def predict(self):
        """
        :return:
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        """
        :param tracks:
        :return:
        """
        if len(tracks) > 0:
            multi_mean = np.asarray([track.mean.copy() for track in tracks])
            multi_covariance = np.asarray([track.covariance for track in tracks])

            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = MCTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

                tracks[i].age += 1

                if tracks[i].time_since_last_update > 0:
                    tracks[i].hit_streak = 0

                tracks[i].time_since_last_update += 1

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: MCTrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1

        self.score = new_track.score

        new_tlwh = new_track.tlwh

        self.time_since_last_update = 0

        self.mean, self.covariance = \
            self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            # if new_track.tlwh[2]*new_track.tlwh[3] >= 600:
            self.update_features(new_track.curr_feat)
            self.update_features_custom(new_track.curr_feat, new_track.score)

        # ----- Update the states
        self.state = TrackState.Tracked
        self.is_activated = True

    def activate(self, kalman_filter, frame_id):
        """
        Start a new track-let: the initial activation
        :param kalman_filter:
        :param frame_id:
        :return:
        """
        self.kalman_filter = kalman_filter

        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)

        # init Kalman filter when activated
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        # ----- init states
        self.track_len = 0
        self.state = TrackState.Tracked

        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        :param new_track:
        :param frame_id:
        :param new_id:
        :return:
        """
        # ----- Kalman update
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xywh(new_track.tlwh))

        if new_track.curr_feat is not None:
            # if new_track.tlwh[2] * new_track.tlwh[3] >= 600:
            # self.update_features(new_track.curr_feat)
            self.update_features_custom(new_track.curr_feat, new_track.score)

        # ----- update track-let states
        self.track_len = 0
        self.frame_id = frame_id
        self.score = new_track.score

        # ----- Update tracking states
        self.state = TrackState.Tracked
        self.is_activated = True

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """
        Get current position in bounding box format
        `(top left x, top left y, width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        # ret[2] *= ret[3]
        # ret[:2] -= ret[2:] * 0.5
        ret[:2] -= ret[2:] * 0.5

        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh2tlbr(tlwh):
        """
        :param tlwh:
        """
        ret = tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xyah(self):
        """
        :return:
        """
        return self.tlwh_to_xyah(self.tlwh)

    def to_xywh(self):
        """
        :return:
        """
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        """
        :param tlbr:
        :return:
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        """
        :param tlwh:
        :return:
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        """
        返回一个对象的 string 格式。
        :return:
        """
        return "OT_({}-{})_({}-{})" \
            .format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


class MCBYTETracker:
    def __init__(self, args, frame_rate=30):
        # define track dicts for multi-class objects
        self.tracked_tracks_dict = defaultdict(list)  # type: dict[int, list[MCTrack]]
        self.lost_tracks_dict = defaultdict(list)  # type: dict[int, list[MCTrack]]
        self.removed_tracks_dict = defaultdict(list)  # type: dict[int, list[MCTrack]]

        self.frame_id = 0
        self.args = args

        self.low_det_thresh = 0.1
        self.high_det_thresh = self.args['track_thresh']  # 0.5
        self.high_match_thresh = self.args['match_thresh']  # 0.8
        # self.high_match_thresh = chi2inv95[4]
        # self.high_match_thresh = 2000
        self.low_match_thresh = 0.5
        # self.low_match_thresh = chi2inv95[8]
        # self.low_match_thresh = 1000
        self.unconfirmed_match_thresh = 0.7
        # self.unconfirmed_match_thresh = chi2inv95[4]
        # self.unconfirmed_match_thresh = 1800
        # self.new_track_thresh = self.high_det_thresh + 0.2
        self.new_track_thresh = 0.5

        # print("Tracker's low det thresh: ", self.low_det_thresh)
        # print("Tracker's high det thresh: ", self.high_det_thresh)
        # print("Tracker's high match thresh: ", self.high_match_thresh)
        # print("Tracker's low match thresh: ", self.low_match_thresh)
        # print("Tracker's unconfirmed match thresh: ", self.unconfirmed_match_thresh)
        # print("Tracker's new track thresh: ", self.new_track_thresh)

        # self.det_thresh = args.track_thresh
        # self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args['track_buffer'])
        self.max_time_lost = self.buffer_size
        # print("Tracker's buffer size: ", self.buffer_size)

        self.kalman_filter = KalmanFilter()

        # Get number of tracking object classes
        self.class_names = args['class_names']
        self.n_classes = args['n_classes']

        self.tracks = []
        self.tracked_tracks = []

        # self.max_age = self.buffer_size
        self.min_hits = 3

        if args['with_reid']:
            self.encoder = FastReIDInterface(args['fast_reid_config'], args['fast_reid_weights'], args['device'])

        # ReID module
        self.proximity_thresh = 0.8
        self.appearance_thresh = 0.25

    def update(self, dets, img):
        # update frame id
        self.frame_id += 1

        # reset track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCTrack.init_id_dict(self.n_classes)

        # gpu -> cpu
        # with torch.no_grad():
        #     dets = dets.cpu().numpy()

        # The current frame 8 tracking states recording
        unconfirmed_tracks_dict = defaultdict(list)
        tracked_tracks_dict = defaultdict(list)
        track_pool_dict = defaultdict(list)
        activated_tracks_dict = defaultdict(list)
        refind_tracks_dict = defaultdict(list)
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        # Even: Start MCMOT
        # Fill box dict and score dict
        bboxes_dict = defaultdict(list)
        scores_dict = defaultdict(list)

        for det in dets:
            x1, y1, x2, y2, score, cls_id = det  # det.size == 6
            if det.size == 7:
                x1, y1, x2, y2, score1, score2, cls_id = det  # det.size == 7
                score = score1 * score2

            bbox = np.array([x1, y1, x2, y2])

            bboxes_dict[int(cls_id)].append(bbox)
            scores_dict[int(cls_id)].append(score)

        # Process each object class
        for cls_id in range(self.n_classes):
            # class boxes
            bboxes = bboxes_dict[cls_id]
            bboxes = np.array(bboxes)

            # class scores
            scores = scores_dict[cls_id]
            scores = np.array(scores)

            # first group indexs
            inds_first = scores > self.high_det_thresh

            # second group indexs
            inds_low = scores > self.low_det_thresh
            inds_high = scores < self.high_det_thresh
            indes_second = np.logical_and(inds_low, inds_high)

            bboxes_first = bboxes[inds_first]
            scores_first = scores[inds_first]

            bboxes_second = bboxes[indes_second]
            scores_second = scores[indes_second]

            '''Extract embeddings '''
            if self.args['with_reid']:
                features_keep = self.encoder.inference(img, bboxes_first)
            else:
                features_keep = [None] * len(bboxes_first)

            if len(bboxes_first) > 0:
                '''Detections'''
                detections_first = [MCTrack(MCTrack.tlbr_to_tlwh(tlbr), s, cls_id, f) for
                                    (tlbr, s, f) in zip(bboxes_first, scores_first, features_keep)]
            else:
                detections_first = []

            ''' Add newly detected tracklets to tracked_tracks'''
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_tracks_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            ''' Step 2: First association, with high score detection boxes'''
            track_pool_dict[cls_id] = joint_stracks(tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id])
            self.tracks = track_pool_dict[cls_id]
            self.tracked_tracks = tracked_tracks_dict[cls_id]

            # Predict the current location with KF
            MCTrack.multi_predict(self.tracks)  # only predict tracked tracks

            # Matching with IOU distance and Hungarian Algorithm
            ious_dists = matching.iou_distance(self.tracks, detections_first)
            # print(ious_dists)
            ious_dists_mask = (ious_dists > self.proximity_thresh)

            # dists = matching.fuse_motion(self.kalman_filter, dists, self.tracks, detections_first, lambda_=0)
            if self.args['with_reid']:

                # BOT-Fusion
                # emb_dists = matching.embedding_distance(track_pool_dict[cls_id], detections_first)
                # emb_dists /= 2.0
                # raw_emb_dists = emb_dists.copy()
                # emb_dists[emb_dists > self.appearance_thresh] = 1.0
                # emb_dists[ious_dists_mask] = 1.0
                # dists = np.minimum(ious_dists, emb_dists)

                # Size-Fusion custom 1
                # emb_dists = matching.embedding_distance(track_pool_dict[cls_id], detections_first)
                # size_dists = matching.size_cost(track_pool_dict[cls_id], detections_first)
                # emb_dists[size_dists < 600] = 1.0
                # emb_dists[ious_dists_mask] = 1.0
                # dists = np.minimum(ious_dists, emb_dists)

                # Size-Fusion custom 2
                # emb_dists = matching.embedding_distance(track_pool_dict[cls_id], detections_first)
                # emb_dists_mask = emb_dists > 0.75
                # size_dists = matching.size_cost(track_pool_dict[cls_id], detections_first)
                # size_dists_mask = size_dists < 600
                # # print(size_dists)
                # emb_dists[emb_dists_mask] = 1.0
                # emb_dists[size_dists_mask] = 1.0
                # emb_dists[ious_dists_mask] = 1.0
                # # print(emb_dists)
                # dists = np.minimum(ious_dists, emb_dists)

                # Size-Fusion custom 3
                emb_dists = matching.embedding_distance(self.tracks, detections_first)
                emb_dists_mask = emb_dists > 0.25
                size_dists = matching.size_cost(self.tracks, detections_first)
                size_dists_mask = size_dists < 600
                # print(size_dists)
                emb_dists[size_dists_mask] = 1.0
                emb_dists[emb_dists_mask] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                # print(emb_dists)
                ious_dists[size_dists >= 600] += 0.3
                dists = np.minimum(ious_dists, emb_dists)

                # IoU making ReID
                # dists = matching.embedding_distance(track_pool_dict[cls_id], detections_first)
                # dists[ious_dists_mask] = 1.0
            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.high_match_thresh)
            # print(matches)

            # process matched pairs between track pool and current frame detection
            for i_tracked, i_det in matches:
                track = self.tracks[i_tracked]
                det = detections_first[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    # print(det.tlwh)
                    # print(track.tlwh)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with low score detection boxes'''
            # association the untrack to the low score detections
            if len(bboxes_second) > 0:
                '''Detections'''
                detections_second = [MCTrack(MCTrack.tlbr_to_tlwh(tlbr), s, cls_id) for
                                     (tlbr, s) in zip(bboxes_second, scores_second)]
            else:
                detections_second = []

            r_tracked_stracks = [self.tracks[i] for i in u_track if self.tracks[i].state == TrackState.Tracked]
            # Matching with IOU distance and Hungarian Algorithm
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            # dists = matching.fuse_motion(self.kalman_filter, dists, r_tracked_stracks, detections_second, lambda_=0)

            matches, u_track, u_detection_second = matching.linear_assignment(
                dists, thresh=self.low_match_thresh)  # thresh=0.5

            for i_tracked, i_det in matches:
                track = r_tracked_stracks[i_tracked]
                det = detections_second[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_tracks_dict[cls_id].append(track)

            # process unmatched tracks
            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    # mark unmatched track as lost track
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections_first = [detections_first[i] for i in u_detection]

            dists = matching.iou_distance(unconfirmed_tracks_dict[cls_id], detections_first)
            # dists = matching.fuse_score(dists, detections_first)
            # dists = matching.fuse_motion(self.kalman_filter, dists, unconfirmed_tracks_dict[cls_id], detections_first, lambda_=0)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists,
                                                                             thresh=self.unconfirmed_match_thresh)  # 0.7
            for i_tracked, i_det in matches:
                track = unconfirmed_tracks_dict[cls_id][i_tracked]
                det = detections_first[i_det]
                track.update(det, self.frame_id)
                activated_tracks_dict[cls_id].append(track)

            for i_track in u_unconfirmed:  # process unconfirmed tracks
                track = unconfirmed_tracks_dict[cls_id][i_track]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)
                # print(f"Class {cls_id}: Remove unconfirmed tracklet {track.track_id}")

            """ Step 4: Init new stracks"""
            for i_new in u_detection:
                track = detections_first[i_new]
                if track.score < self.new_track_thresh:
                    continue

                # tracked but not activated: activate do not set 'is_activated' to be True
                # if fr_id > 1, tracked but not activated
                track.activate(self.kalman_filter, self.frame_id)

                # activated_tracks_dict may contain track with 'is_activated' False
                activated_tracks_dict[cls_id].append(track)

            """ Step 5: Update state"""
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)
                    # print(f"Class {cls_id}: Remove tracklet {track.track_id}")

            """Post processing"""
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = joint_stracks(self.tracked_tracks_dict[cls_id],
                                                             activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = joint_stracks(self.tracked_tracks_dict[cls_id],
                                                             refind_tracks_dict[cls_id])

            self.lost_tracks_dict[cls_id] = sub_stracks(self.lost_tracks_dict[cls_id],
                                                        self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_stracks(self.lost_tracks_dict[cls_id],
                                                        self.removed_tracks_dict[cls_id])

            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_stracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id] if track.is_activated]
            # output_tracks_dict[cls_id].extend(self.lost_tracks_dict[cls_id])
        # Return final online target of the frame
        return output_tracks_dict
        # End MCMOT


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
