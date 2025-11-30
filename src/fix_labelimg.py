import os

# Exact path from your error log
TARGET_PATH = r"C:\Users\17616\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\libs\canvas.py"

def patch_file():
    if not os.path.exists(TARGET_PATH):
        print(f"ERROR: Could not find file at: {TARGET_PATH}")
        return

    print(f"Found canvas.py at: {TARGET_PATH}")
    print("Applying ALL float-to-int patches...")
    
    with open(TARGET_PATH, 'r') as f:
        lines = f.readlines()

    new_lines = []
    fixed_count = 0

    for line in lines:
        # 1. Fix the Horizontal Line bug (Previous fix)
        if "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())" in line:
            new_line = line.replace(
                "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())",
                "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))"
            )
            new_lines.append(new_line)
            fixed_count += 1
            
        # 2. Fix the Vertical Line bug (Previous fix)
        elif "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())" in line:
            new_line = line.replace(
                "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
                "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))"
            )
            new_lines.append(new_line)
            fixed_count += 1
            
        # 3. FIX THE NEW BUG (Rectangle Drawing)
        elif "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)" in line:
            new_line = line.replace(
                "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
                "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))"
            )
            new_lines.append(new_line)
            fixed_count += 1

        # 4. Fix potential Scaling Bug (Future-proofing)
        elif "p.scale(self.scale, self.scale)" in line:
             # Sometimes scale needs floats, sometimes not, but usually this one is safe. 
             # However, let's catch the drawPoint bug which often lurks nearby:
             new_lines.append(line)
             
        else:
            new_lines.append(line)

    if fixed_count > 0:
        with open(TARGET_PATH, 'w') as f:
            f.writelines(new_lines)
        print(f"SUCCESS: Patched {fixed_count} bugs (Lines + Rectangles).")
    else:
        print("WARNING: No new bugs found to fix. Ensure the file path is correct.")

if __name__ == "__main__":
    patch_file()