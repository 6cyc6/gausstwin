import cv2


class PromptTool:
    def __init__(self, image_path: str, obj_name: str = "Object"):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        self.clone = self.image.copy()
        self.obj_name = obj_name
        self.window_title = f"Set prompts for {self.obj_name}"

        # State
        self.positive_points = []
        self.negative_points = []
        self.bbox = []
        self.drawing_bbox = False
        self.bbox_start = None
        self.negative_mode = False

    def draw_overlay(self):
        img = self.clone.copy()

        for pt in self.positive_points:
            cv2.circle(img, pt, 5, (0, 255, 0), -1)  # Green
        for pt in self.negative_points:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)  # Red

        if len(self.bbox) == 4:
            cv2.rectangle(img, (self.bbox[0], self.bbox[1]),
                          (self.bbox[2], self.bbox[3]), (255, 0, 0), 2)  # Blue

        cv2.imshow(self.window_title, img)
        self.image = img
        

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing_bbox:
                self.bbox_start = (x, y)
            else:
                if self.negative_mode:
                    print(f"🔴 Added negative point: ({x}, {y})")
                    self.negative_points.append((x, y))
                else:
                    print(f"🟢 Added positive point: ({x}, {y})")
                    self.positive_points.append((x, y))
                self.draw_overlay()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_bbox and self.bbox_start:
                temp_img = self.clone.copy()
                cv2.rectangle(temp_img, self.bbox_start, (x, y), (255, 0, 0), 2)
                for pt in self.positive_points:
                    cv2.circle(temp_img, pt, 5, (0, 255, 0), -1)
                for pt in self.negative_points:
                    cv2.circle(temp_img, pt, 5, (0, 0, 255), -1)
                cv2.imshow(self.window_title, temp_img)

        elif event == cv2.EVENT_LBUTTONUP and self.bbox_start:
            x1, y1 = self.bbox_start
            x2, y2 = x, y
            self.bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            self.drawing_bbox = False
            self.bbox_start = None
            print(f"📦 Bounding box selected: {self.bbox}")
            self.draw_overlay()
            

    def run(self):
        cv2.namedWindow(self.window_title)
        cv2.setMouseCallback(self.window_title, self.mouse_callback)
        self.draw_overlay()

        print(
            f"💡 Controls for '{self.obj_name}':\n"
            "Add points:\n"
            "- Left click: Add point (🟢 positive / 🔴 negative)\n"
            "- Press 'n': Toggle to negative/positive mode\n"
            "- Press 'b': Undo last point\n"
            "Draw bounding box:\n"
            "- Press 'r': Start drawing bounding box (click & drag)\n"
            "- Draw a new bounding box will overwrite the previous one\n"
            "Save and finish:\n"
            "- Press 'Enter': Finish and print results\n"
            "- Press 'q': Quit without saving\n"
        )

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('b'):
                if self.positive_points:
                    removed = self.positive_points.pop()
                    print(f"↩️ Removed positive point: {removed}")
                elif self.negative_points:
                    removed = self.negative_points.pop()
                    print(f"↩️ Removed negative point: {removed}")
                self.draw_overlay()

            elif key == ord('r'):
                self.drawing_bbox = True
                self.bbox_start = None
                print("🖱️ Draw bounding box: click and drag.")

            elif key == ord('n'):
                self.negative_mode = not self.negative_mode
                mode = "🔴 Negative Mode" if self.negative_mode else "🟢 Positive Mode"
                print(f"✴️ Switched to: {mode}")

            elif key == 13:  # Enter
                break

            elif key == ord('q'):
                print("❌ Quit without saving.")
                self.positive_points, self.negative_points, self.bbox = [], [], []
                break

        cv2.destroyAllWindows()

        # Final outputs
        print(f"\n✅ Positive points: {self.positive_points}")
        print(f"❌ Negative points: {self.negative_points}")
        print(f"📦 Bounding box: {self.bbox}")
        
        # Combine points and labels
        all_points = self.positive_points + self.negative_points
        labels = [1] * len(self.positive_points) + [0] * len(self.negative_points)
        
        return all_points, labels, self.bbox
    