import time
import math
import struct
from collections import deque

import pygame


class AlertManager:

    WINDOW_SECONDS = 90.0

    def reset_all(self):

        self.yawn_events.clear()
        self.drowsiness_events.clear()

        self.previous_yawn = False
        self.previous_drowsiness = False
        self.previous_distraction = False

        self.fatigue_active = False

        if self.audio_enabled:
            self.channel.stop()

        print("All alert history and counters reset.")

    def __init__(self):

        # Event history
        self.yawn_events = deque()
        self.drowsiness_events = deque()

        # Track previous states to detect new episodes
        self.previous_yawn = False
        self.previous_drowsiness = False
        self.previous_distraction = False

        # Combined alert state
        self.fatigue_active = False

        # Audio configuration
        self.audio_enabled = False
        self.channel = None
        self.sounds = {}

        self.cooldowns = {
            "yawn": 1.0,
            "drowsiness": 2.0,
            "distraction": 2.0,
            "fatigue": 15.0
        }

        self.last_played = {
            name: 0.0 for name in self.cooldowns
        }

        self.initialize_audio()

    # --------------------------------------------------
    # AUDIO INITIALIZATION
    # --------------------------------------------------

    def initialize_audio(self):

        try:
            pygame.mixer.init(
                frequency=22050,
                size=-16,
                channels=1,
                buffer=512
            )

            pygame.mixer.set_num_channels(1)
            self.channel = pygame.mixer.Channel(0)

            self.sounds = {
                "yawn": self.create_sound([
                    (880, 0.15)
                ]),

                "drowsiness": self.create_sound([
                    (660, 0.18),
                    (None, 0.08),
                    (660, 0.18)
                ]),

                "distraction": self.create_sound([
                    (1000, 0.12),
                    (None, 0.06),
                    (750, 0.12)
                ]),

                "fatigue": self.create_sound([
                    (500, 0.22),
                    (None, 0.08),
                    (700, 0.22),
                    (None, 0.08),
                    (900, 0.35)
                ])
            }

            self.audio_enabled = True
            print("Audio Alert System Initialized Successfully!")

        except pygame.error as error:

            self.audio_enabled = False
            print("Warning: Audio initialization failed:", error)

    # --------------------------------------------------
    # GENERATE SOUND WITHOUT EXTERNAL AUDIO FILES
    # --------------------------------------------------

    def create_sound(self, pattern):

        sample_rate = 22050
        volume = 0.25

        audio_data = bytearray()

        for frequency, duration in pattern:

            sample_count = int(sample_rate * duration)

            if frequency is None:

                audio_data.extend(
                    b"\x00\x00" * sample_count
                )

                continue

            for i in range(sample_count):

                t = i / sample_rate

                fade_in = min(1.0, i / (sample_rate * 0.01))
                fade_out = min(
                    1.0,
                    (sample_count - i) / (sample_rate * 0.02)
                )

                envelope = min(fade_in, fade_out)

                sample = int(
                    32767
                    * volume
                    * envelope
                    * math.sin(2 * math.pi * frequency * t)
                )

                audio_data.extend(
                    struct.pack("<h", sample)
                )

        return pygame.mixer.Sound(
            buffer=bytes(audio_data)
        )

    # --------------------------------------------------
    # PLAY AUDIO
    # --------------------------------------------------

    def play_alert(self, alert_type, current_time):

        if not self.audio_enabled:
            return

        cooldown = self.cooldowns[alert_type]

        if current_time - self.last_played[alert_type] < cooldown:
            return

        self.channel.play(self.sounds[alert_type])

        self.last_played[alert_type] = current_time

    # --------------------------------------------------
    # REMOVE EXPIRED EVENTS
    # --------------------------------------------------

    def clean_old_events(self, current_time):

        while (
            self.yawn_events
            and current_time - self.yawn_events[0] > self.WINDOW_SECONDS
        ):
            self.yawn_events.popleft()

        while (
            self.drowsiness_events
            and current_time - self.drowsiness_events[0] > self.WINDOW_SECONDS
        ):
            self.drowsiness_events.popleft()

    # --------------------------------------------------
    # COMBINED FATIGUE DETECTION
    # --------------------------------------------------

    def check_fatigue(self):

        yawns = len(self.yawn_events)
        drowsiness = len(self.drowsiness_events)

        fatigue_detected = (
            yawns >= 3
            or drowsiness >= 3
            or (
                yawns >= 1
                and drowsiness >= 1
                and yawns + drowsiness >= 3
            )
        )

        return fatigue_detected

    # --------------------------------------------------
    # MAIN UPDATE FUNCTION
    # --------------------------------------------------

    def update(
        self,
        drowsiness_alert=False,
        yawn_alert=False,
        distraction_alert=False
    ):

        current_time = time.monotonic()

        # Detect new alert episodes
        new_yawn = (
            yawn_alert and not self.previous_yawn
        )

        new_drowsiness = (
            drowsiness_alert and not self.previous_drowsiness
        )

        new_distraction = (
            distraction_alert and not self.previous_distraction
        )

        # Update previous states
        self.previous_yawn = bool(yawn_alert)
        self.previous_drowsiness = bool(drowsiness_alert)
        self.previous_distraction = bool(distraction_alert)

        # Record new events
        if new_yawn:

            self.yawn_events.append(current_time)
            print("New Yawn Event Detected")

        if new_drowsiness:

            self.drowsiness_events.append(current_time)
            print("New Drowsiness Event Detected")

        # Remove events older than 90 seconds
        self.clean_old_events(current_time)

        # Evaluate combined fatigue
        fatigue_detected = self.check_fatigue()

        # Detect transition into combined fatigue state
        new_fatigue_alert = (
            fatigue_detected and not self.fatigue_active
        )

        self.fatigue_active = fatigue_detected

        # Audio priority:
        # Combined Fatigue > Drowsiness > Distraction > Yawn

        if new_fatigue_alert:

            print("COMBINED FATIGUE ALERT!")
            self.play_alert("fatigue", current_time)

        elif new_drowsiness:

            self.play_alert("drowsiness", current_time)

        elif new_distraction:

            self.play_alert("distraction", current_time)

        elif new_yawn:

            self.play_alert("yawn", current_time)

        return {
            "fatigue_alert": fatigue_detected,
            "yawn_count": len(self.yawn_events),
            "drowsiness_count": len(self.drowsiness_events)
        }

    # --------------------------------------------------
    # RESET ACTIVE STATES WHEN FACE DISAPPEARS
    # --------------------------------------------------

    def reset_active_states(self):

        self.previous_yawn = False
        self.previous_drowsiness = False
        self.previous_distraction = False

    # --------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------

    def close(self):

        if self.audio_enabled:

            self.channel.stop()
            pygame.mixer.quit()

            self.audio_enabled = False