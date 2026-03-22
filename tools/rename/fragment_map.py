import csv
import os

bot_root_abs = "C:\\bot"
map_fragments = []
with open("path_map.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")
    for old_abs, new_abs in reader:
        old_rel = os.path.relpath(old_abs, bot_root_abs)
        new_rel = os.path.relpath(new_abs, bot_root_abs)
        old_parts = tuple(old_rel.split(os.sep))
        new_parts = tuple(new_rel.split(os.sep))
        map_fragments.append((old_parts, new_parts))
with open("fragment_map.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    for old_parts, new_parts in map_fragments:
        writer.writerow([",".join(old_parts), ",".join(new_parts)])
