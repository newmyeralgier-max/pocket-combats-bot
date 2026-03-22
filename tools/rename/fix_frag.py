import ast
import csv
import os

import astor

FRAGMENTS_FILE = "fragment_map.csv"
PROJECT_ROOT = "C:\\bot"
fragment_map = {}
with open(FRAGMENTS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")
    for old, new in reader:
        old_tuple = tuple(s.strip() for s in old.split(",") if s.strip())
        new_tuple = tuple(s.strip() for s in new.split(",") if s.strip())
        fragment_map[old_tuple] = new_tuple


class JoinRewriter(ast.NodeTransformer):

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            if len(node.args) >= 2:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Call) and getattr(first_arg.func, "id", "") == "bot_root":
                    str_args = []
                    for arg in node.args[1:]:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            str_args.append(arg.value)
                        else:
                            break
                    old_tuple = tuple(str_args)
                    if old_tuple in fragment_map:
                        new_parts = fragment_map[old_tuple]
                        print(f"[{file_path}] {old_tuple}  →  {new_parts}")
                        node.args = [first_arg] + [ast.Constant(s) for s in new_parts]
        return node


def process_file(path):
    global file_path
    file_path = path
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"[SKIP:SyntaxError] {path} — {e}")
        return
    new_tree = JoinRewriter().visit(tree)
    ast.fix_missing_locations(new_tree)
    new_src = astor.to_source(new_tree)
    if new_src != src:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_src)
        print(f"[UPDATED] {path}")


for root, _, files in os.walk(PROJECT_ROOT):
    for file in files:
        if file.endswith(".py"):
            process_file(os.path.join(root, file))
