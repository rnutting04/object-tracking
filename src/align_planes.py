import cv2
import numpy as np

# Configuration variables
CANVAS_SIZE = 1500

# Global variables used to track state
offset_x = 0.0
offset_y = 0.0
scale = 100.0
dragging_pos = False
last_mouse_pos = (0, 0)

def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events to move the Red Box.
    """
    global dragging_pos, last_mouse_pos, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging_pos = True
        last_mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_pos = False
    
    if event == cv2.EVENT_MOUSEMOVE:
        if dragging_pos:
            dx = x - last_mouse_pos[0]
            dy = y - last_mouse_pos[1]
            last_mouse_pos = (x, y)

            # Update offsets based on zoom scale
            offset_x += dx / scale
            offset_y -= dy / scale # Invert Y for world coords

def world_to_screen(wx, wy, cx, cy):
    """
    Convert world meters to screen pixels using current scale
    """
    sx = int(cx + wx * scale)

    sy = int(cy - wy * scale) # Flip Y for standard image coords

    return (sx, sy)

def get_rect_corners(w, h, off_x=0, off_y=0):
    """
    Calculates the 4 corners of a rectangle in World Coordinates
    """
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    final = corners + np.array([off_x, off_y])

    return final

def draw_scene(w1, h1, w2, h2):
    """
    Draws the grid, anchor plane (blue), and floater plane (red).
    """
    # Initalize canvas
    img = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    cx, cy = CANVAS_SIZE // 2, CANVAS_SIZE // 2

    # Draw Grid
    grid_color = (30, 30, 30)
    grid_meters = 30
    
    # Vertical lines
    for i in range(-grid_meters, grid_meters):
        sx, _ = world_to_screen(i, 0, cx, cy)
        if 0 <= sx < CANVAS_SIZE:
            cv2.line(img, (sx, 0), (sx, CANVAS_SIZE), grid_color, 1)
    
    # Horizontal lines
    for i in range(-grid_meters, grid_meters):
        _, sy = world_to_screen(0, i, cx, cy)
        if 0 <= sy < CANVAS_SIZE:
            cv2.line(img, (0, sy), (CANVAS_SIZE, sy), grid_color, 1)

    # Draw Anchor Plane
    anchor_corners = get_rect_corners(w1, h1, 0, 0)
    anchor_pts = [world_to_screen(p[0], p[1], cx, cy) for p in anchor_corners]
    cv2.polylines(img, [np.array(anchor_pts)], True, (255, 100, 0), 2)
    
    # Label Anchor
    ox, oy = anchor_pts[0]
    cv2.putText(img, "Anchor Origin (0,0)", (ox - 20, oy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)

    # Draw Floater Plane
    floater_corners = get_rect_corners(w2, h2, offset_x, offset_y)
    floater_pts = [world_to_screen(p[0], p[1], cx, cy) for p in floater_corners]
    cv2.polylines(img, [np.array(floater_pts)], True, (0, 100, 255), 2)
    
    # Draw Line connecting origins to visualize offset
    f_ox, f_oy = floater_pts[0]
    cv2.line(img, (ox, oy), (f_ox, f_oy), (100, 100, 100), 1)

    # UI Text
    info = [
        f"L-Click Drag: Move Red Box",
        f"+/- Keys: Zoom (Scale: {scale:.0f})",
        f"Enter: Save & Finish",
        f"-------------------",
        f"Offset X: {offset_x:.4f} m",
        f"Offset Y: {offset_y:.4f} m",
    ]

    for i, line in enumerate(info):
        cv2.putText(img, line, (10, 35 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return img

def get_alignment_offset(w1, h1, w2, h2):
    """
    Opens the GUI tool to interactively align the two planes.
    """
    global offset_x, offset_y, scale
    
    # Reset state every time the function is called
    offset_x, offset_y = 0.0, 0.0
    scale = 100.0

    window_name = "Align Planes - Press Enter to Confirm"
    
    # Enable Resizable Window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 800) 
    
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- Align Planes Tool ---")
    print("Adjust the Red Box to match the Blue Box relative position.")
    print("Press ENTER to return the calculated offset.")

    while True:
        img = draw_scene(w1, h1, w2, h2)
        cv2.imshow(window_name, img)
        
        key = cv2.waitKey(10)
        
        if key == 13: # Enter Key
            break
        elif key == ord('q'):
            print("Quit pressed. Returning (0,0).")
            offset_x, offset_y = 0.0, 0.0
            break
        elif key == ord('=') or key == ord('+'):
            scale += 5
        elif key == ord('-') or key == ord('_'):
            scale = max(5, scale - 5)

    cv2.destroyAllWindows()
    return offset_x, offset_y

if __name__ == "__main__":
    """
    Script used to run the function independently
    """
    W1, H1 = 7.9, 1.0  
    W2, H2 = 1.0, 4.0
    
    final_x, final_y = get_alignment_offset(W1, H1, W2, H2)
    
    print("\n" + "="*30)
    print(f"Returned offset: X={final_x:.4f}, Y={final_y:.4f}")
    print("="*30 + "\n")