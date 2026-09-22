"""Shared construction of BehaveAI motion images from video frames."""

import cv2
import numpy as np


def motion_warmup_samples(strategy, exp_a, exp_b, history_frames=3):
	"""Return the sampled-frame window needed to reproduce a motion image."""
	history_frames = int(history_frames)
	if not 1 <= history_frames <= 3:
		raise ValueError("motion_history_frames must be between 1 and 3")
	window = history_frames + 1
	if strategy == 'exponential' and history_frames > 1:
		if exp_a > 0.2 or exp_b > 0.2: window = max(window, 5)
		if exp_a > 0.5 or exp_b > 0.5: window = max(window, 10)
		if exp_a > 0.7 or exp_b > 0.7: window = max(window, 15)
		if exp_a > 0.8 or exp_b > 0.8: window = max(window, 20)
		if exp_a > 0.9 or exp_b > 0.9: window = max(window, 45)
	return window


class MotionProcessor:
	"""Incrementally build motion images while keeping sampling independent of output skipping."""

	def __init__(self, history_frames=3, frame_gap=1, strategy='exponential',
				 exp_a=0.5, exp_b=0.8, lum_weight=0.7,
				 rgb_multipliers=(2, 2, 2), motion_threshold=0,
				 chromatic_tail_only=False):
		self.history_frames = int(history_frames)
		self.frame_gap = int(frame_gap)
		if not 1 <= self.history_frames <= 3:
			raise ValueError("motion_history_frames must be between 1 and 3")
		if self.frame_gap < 1:
			raise ValueError("motion_frame_gap must be at least 1")
		if strategy not in ('sequential', 'exponential'):
			raise ValueError("strategy must be sequential or exponential")
		self.strategy = strategy
		self.exp_a = float(exp_a)
		self.exp_b = float(exp_b)
		self.lum_weight = float(lum_weight)
		self.rgb_multipliers = tuple(float(x) for x in rgb_multipliers)
		self.motion_threshold = float(motion_threshold)
		self.chromatic_tail_only = bool(chromatic_tail_only)
		self.reset()

	def reset(self):
		self._source_index = 0
		self._previous_by_phase = {}
		self.last_motion_image = None

	def update(self, frame):
		"""Consume one source frame and compare it with its same-gap history."""
		phase = self._source_index % self.frame_gap
		self._source_index += 1
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		previous = self._previous_by_phase.get(phase)
		if previous is None:
			self._previous_by_phase[phase] = [gray.copy() for _ in range(self.history_frames)]
			return None

		diffs = [cv2.absdiff(item, gray) for item in previous]
		if self.strategy == 'exponential':
			previous[0] = gray
			if self.history_frames > 1:
				previous[1] = cv2.addWeighted(previous[1], self.exp_a, gray, 1 - self.exp_a, 0)
			if self.history_frames > 2:
				previous[2] = cv2.addWeighted(previous[2], self.exp_b, gray, 1 - self.exp_b, 0)
		else:
			for index in range(self.history_frames - 1, 0, -1):
				previous[index] = previous[index - 1]
			previous[0] = gray

		if self.history_frames == 1:
			diffs = [diffs[0], diffs[0], diffs[0]]
		elif self.history_frames == 2:
			diffs = [diffs[0], diffs[1], diffs[1]]

		# A chromatic tail requires at least two different time histories.
		if self.chromatic_tail_only and self.history_frames > 1:
			tb = cv2.subtract(diffs[0], diffs[1])
			tr = cv2.subtract(diffs[2], diffs[1])
			tg = cv2.subtract(diffs[1], diffs[0])
			channels = (tb, tg, tr)
		else:
			channels = tuple(diffs)

		blue = cv2.addWeighted(gray, self.lum_weight, channels[0], self.rgb_multipliers[2], self.motion_threshold)
		green = cv2.addWeighted(gray, self.lum_weight, channels[1], self.rgb_multipliers[1], self.motion_threshold)
		red = cv2.addWeighted(gray, self.lum_weight, channels[2], self.rgb_multipliers[0], self.motion_threshold)
		self.last_motion_image = cv2.merge((blue, green, red)).astype(np.uint8)
		return self.last_motion_image


def build_motion_image(frames, **kwargs):
	"""Build the final motion image from a contiguous iterable of source frames."""
	processor = MotionProcessor(**kwargs)
	result = None
	for frame in frames:
		candidate = processor.update(frame)
		if candidate is not None:
			result = candidate
	return result
