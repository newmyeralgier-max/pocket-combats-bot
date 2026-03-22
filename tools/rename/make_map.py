import csv


OLD_FILE = 'old_path.txt'
NEW_FILE = 'new_file_list.txt'
OUT_FILE = 'path_map.csv'


def read_paths(txt_file):
    paths = []
    with open(txt_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('fullname'):
                continue
            paths.append(line)
    return paths


def to_forward_slash(p):
    return p.replace('\\', '/')


def to_relative(p):
    fs = to_forward_slash(p)
    if fs.lower().startswith('c:/bot/'):
        return fs[len('C:/bot/'):]
    return fs


def main():
    old_paths = read_paths(OLD_FILE)
    new_paths = read_paths(NEW_FILE)
    if len(old_paths) != len(new_paths):
        print(
            f'[WARN] Кол-во строк отличается: old={len(old_paths)}, new={len(new_paths)}'
            )
    rows = []
    for old, new in zip(old_paths, new_paths):
        if not new or new.upper() == 'NOT_FOUND':
            continue
        rows.append([old, new])
        rows.append([to_forward_slash(old), to_forward_slash(new)])
        rel_old = to_relative(old)
        rel_new = to_relative(new)
        if rel_old != old:
            rows.append([rel_old, rel_new])
    unique_rows = []
    seen = set()
    for old, new in rows:
        key = old, new
        if key not in seen:
            seen.add(key)
            unique_rows.append((old, new))
    with open(OUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for old, new in unique_rows:
            writer.writerow([old, new])
    print(f'[OK] Готово: {OUT_FILE} ({len(unique_rows)} строк)')
# Get-ChildItem -Recurse -File | Select-Object FullName | Out-File "C:\bot\new_file_list.txt" -Encoding UTF8
# Get-ChildItem -Recurse -File | Select-Object FullName | Out-File "C:\Users\Максим\AppData\new_file_list.txt" -Encoding UTF8


if __name__ == '__main__':
    main()
