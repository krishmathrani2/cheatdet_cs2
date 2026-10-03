from demoparser2 import DemoParser

parser = DemoParser("match730_003845848209444307084_2046226831_274.dem")

header = parser.parse_header()
print(header)  # map name etc.

ticks = parser.parse_ticks(["X", "Y", "Z", "pitch", "yaw", "health", "team_num"])
print(ticks.shape)
print(ticks["steamid"].nunique())   # should be 10 (possibly a few more if people reconnected)
print(ticks.head())