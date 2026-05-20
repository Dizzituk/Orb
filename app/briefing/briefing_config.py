  1:   1: from __future__ import annotations
  2:   2: 
  3:   3: import json
  4:   4: import logging
  5:   5: import os
  6:   6: from dataclasses import dataclass, field
  7:   7: from enum import Enum
  8:   8: from typing import Optional
  9:   9: 
 10:  10:  11: 
 11:  12: logger = logging.getLogger(__name__)
 12:  13: 
 13:  14: 
 14:  15: class BriefingFrequency(str, Enum):
 15:  16:     DAILY = "daily"
 16:  17:     WEEKLY = "weekly"
 17:  18: 
 18:  19: 
 19:  20: @dataclass
 20:  21: class TopicConfig:
 21:  22:     name: str
 22:  23:     key: str
 23:  24:     enabled: bool = True
 24:  25:     priority: int = 1
 25:  26:     search_queries: list = field(default_factory=list)
 26:  27:     max_stories: int = 5
 27:  28:     freshness_hint: str = "today"
 28:  29:     description: str = ""
 29:  30:     astra_relevant: bool = False
 30:  31: 
 31:  32: 
 32:  33: _DEFAULT_TOPICS: list[TopicConfig] = [
 33:  34:     TopicConfig(
 34:  35:         name="Financial Markets & Crypto",
 35:  36:         key="finance",
 36:  37:         priority=1,
 37:  38:         search_queries=[
 38:  39:             "financial markets today summary",
 39:  40:             "cryptocurrency news today",
 40:  41:             "TAO bittensor price news",
 41:  42:             "FTSE 100 today",
 42:  43:             "global economy news today",
 43:  44:         ],
 44:  45:         max_stories=5,
 45:  46:         freshness_hint="today",
 46:  47:         description="Markets, crypto, and economic indicators",
 47:  48:     ),
 48:  49:     TopicConfig(
 49:  50:         name="AI & Technology",
 50:  51:         key="ai_tech",
 51:  52:         priority=2,
 52:  53:         search_queries=[
 53:  54:             "artificial intelligence news today",
 54:  55:             "new AI model release announcement",
 55:  56:             "AI technology breakthrough",
 56:  57:             "LLM benchmark leaderboard latest",
 57:  58:         ],
 58:  59:         max_stories=5,
 59:  60:         freshness_hint="this week",
 60:  61:         description="AI developments, model releases, and tech industry moves",
 61:  62:         astra_relevant=True,
 62:  63:     ),
 63:  64:     TopicConfig(
 64:  65:         name="UK & World Affairs",
 65:  66:         key="world_affairs",
 66:  67:         priority=3,
 67:  68:         search_queries=[
 68:  69:             "UK news today headlines",
 69:  70:             "world news today top stories",
 70:  71:             "UK economy news today",
 71:  72:         ],
 72:  73:         max_stories=4,
 73:  74:         freshness_hint="today",
 74:  75:         description="UK domestic and international headlines",
 75:  76:     ),
 76:  77:     TopicConfig(
 77:  78:         name="Geopolitics",
 78:  79:         key="geopolitics",
 79:  80:         priority=4,
 80:  81:         search_queries=[
 81:  82:             "geopolitics news today",
 82:  83:             "international relations latest",
 83:  84:             "global conflict update",
 84:  85:         ],
 85:  86:         max_stories=3,
 86:  87:         freshness_hint="today",
 87:  88:         description="International relations, conflicts, and diplomacy",
 88:  89:     ),
 89:  90:     TopicConfig(
 90:  91:         name="Surf & Weather",
 91:  92:         key="surf_weather",
 92:  93:         priority=5,
 93:  94:         search_queries=[
 94:  95:             "surf forecast Devon Cornwall today",
 95:  96:             "Plymouth weather forecast",
 96:  97:             "Portugal surf forecast Nazare Peniche",
 97:  98:         ],
 98:  99:         max_stories=3,
 99: 100:         freshness_hint="today",
100: 101:         description="Local and Portugal surf conditions and weather",
101: 102:     ),
102: 103: ]
103: 104: 
104: 105: 
105: 106: def _load_topics_from_env() -> Optional[list[TopicConfig]]:
106: 107:     raw = os.getenv("BRIEFING_TOPICS_JSON", "").strip()
107: 108:     if not raw:
108: 109:         return None
109: 110:     try:
110: 111:         data = json.loads(raw)
111: 112:         return [TopicConfig(**item) for item in data]
112: 113:     except Exception as exc:
113: 114:         logger.warning("[briefing_config] Failed to parse BRIEFING_TOPICS_JSON: %s", exc)
114: 115:         return None
115: 116: 
116: 117: 
117: 118: def get_topics(profile: str = "default") -> list[TopicConfig]:
118: 119:     if profile != "default":
119: 120:         from app.briefing.news_profiles import get_profile_topics
120: 121: 
121: 122:         profile_topics = get_profile_topics(profile)
120: 121:         if profile_topics:
121: 122:             return profile_topics
122: 123:     env_topics = _load_topics_from_env()
123: 124:     topics = env_topics if env_topics is not None else _DEFAULT_TOPICS
124: 125:     return sorted([topic for topic in topics if topic.enabled], key=lambda topic: topic.priority)
125: 126: 
126: 127: 
127: 128: @dataclass
128: 129: class ScheduleConfig:
129: 130:     daily_hour: int = 6
130: 131:     daily_minute: int = 0
131: 132:     weekly_day: int = 0
132: 133:     weekly_hour: int = 7
133: 134:     weekly_minute: int = 0
134: 135:     auto_generate: bool = True
135: 136:     audio_enabled: bool = True
136: 137: 
137: 138: 
138: 139: @dataclass
139: 140: class VoiceConfig:
140: 141:     voice_headlines: str = "en-GB-Chirp3-HD-Achird"
141: 142:     voice_analysis: str = "en-GB-Chirp3-HD-Fenrir"
142: 143:     speed: float = 1.0
143: 144:     pause_between_stories_ms: int = 800
144: 145:     pause_between_sections_ms: int = 1200
145: 146: 
146: 147: 
147: 148: def get_schedule() -> ScheduleConfig:
148: 149:     return ScheduleConfig(
149: 150:         daily_hour=int(os.getenv("BRIEFING_DAILY_HOUR", "6")),
150: 151:         daily_minute=int(os.getenv("BRIEFING_DAILY_MINUTE", "0")),
151: 152:         weekly_day=int(os.getenv("BRIEFING_WEEKLY_DAY", "0")),
152: 153:         weekly_hour=int(os.getenv("BRIEFING_WEEKLY_HOUR", "7")),
153: 154:         weekly_minute=int(os.getenv("BRIEFING_WEEKLY_MINUTE", "0")),
154: 155:         auto_generate=os.getenv("BRIEFING_AUTO_GENERATE", "true").lower() in ("true", "1"),
155: 156:         audio_enabled=os.getenv("BRIEFING_AUDIO_ENABLED", "true").lower() in ("true", "1"),
156: 157:     )
157: 158: 
158: 159: 
159: 160: def get_voice_config() -> VoiceConfig:
160: 161:     return VoiceConfig(
161: 162:         voice_headlines=os.getenv("BRIEFING_VOICE_HEADLINES", "en-GB-Chirp3-HD-Achird"),
162: 163:         voice_analysis=os.getenv("BRIEFING_VOICE_ANALYSIS", "en-GB-Chirp3-HD-Fenrir"),
163: 164:         speed=float(os.getenv("BRIEFING_VOICE_SPEED", "1.0")),
164: 165:     )
165: 166: 
166: 167: 
167: 168: __all__ = [
168: 169:     "BriefingFrequency",
169: 170:     "TopicConfig",
170: 171:     "ScheduleConfig",
171: 172:     "VoiceConfig",
172: 173:     "get_topics",
173: 174:     "get_schedule",
174: 175:     "get_voice_config",
175: 176: ]