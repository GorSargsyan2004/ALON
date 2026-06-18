from pathlib import Path

from services.tools.filesystem.policy import DESKTOP_ROOT, DOCUMENTS_ROOT, resolve_safe, assert_read_allowed, assert_write_allowed, assert_exec_allowed
from services.tools.filesystem import fs_tool
from services.tools.executor.exec_tool import run_command


def main():
    desktop = DESKTOP_ROOT
    documents = DOCUMENTS_ROOT

    print("Desktop:", desktop)
    print("Documents:", documents)

    # Read allowed in Documents
    assert_read_allowed(documents)
    print("Read allowed in Documents: OK")

    # Write denied in Documents
    try:
        fs_tool.write_file(str(documents / "denied_test.txt"), "nope")
        print("Write in Documents: FAILED (should be denied)")
    except Exception as e:
        print("Write in Documents: OK (denied)")

    # Write allowed in Desktop
    fs_tool.write_file(str(desktop / "allowed_test.txt"), "ok")
    print("Write in Desktop: OK")

    # Exec denied outside Desktop
    try:
        run_command(["python", "-c", "print('hi')"], cwd=str(documents), timeout_sec=5, mode="windows")
        print("Exec in Documents: FAILED (should be denied)")
    except Exception:
        print("Exec in Documents: OK (denied)")

    # Exec allowed in Desktop
    out = run_command(["python", "-c", "print('hi')"], cwd=str(desktop), timeout_sec=5, mode="windows")
    print("Exec in Desktop: OK", out.get("stdout").strip())


if __name__ == "__main__":
    main()
