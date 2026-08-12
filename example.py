import glob
import os
import cv2
import numpy as np
from ultralytics import YOLO


class CarDamageDetector:

  def __init__(
      self,
      stage1_model="yolov8x-seg.pt",
      stage2_model="best.pt",
      car_class_ids=(2, 5, 7),
  ):
    self.m1 = YOLO(stage1_model)
    self.m2 = YOLO(stage2_model)
    self.car_ids = car_class_ids

  def get_car_mask(self, img):
    h, w = img.shape[:2]
    car_mask = np.zeros((h, w), dtype=np.uint8)
    res = self.m1.predict(source=img, conf=0.4, verbose=False)[0]

    if res.masks is not None:
      for box, mask_data in zip(res.boxes, res.masks.data):
        if int(box.cls[0]) in self.car_ids:
          mask = cv2.resize(
              mask_data.cpu().numpy().astype(np.uint8),
              (w, h),
              interpolation=cv2.INTER_NEAREST,
          )
          car_mask = cv2.bitwise_or(car_mask, mask)
    return car_mask

  def check_canny_density(
      self, roi, low_thresh=30, high_thresh=120, density_thresh=0.015
  ):
    if roi.size == 0:
      return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), low_thresh, high_thresh)
    return (np.count_nonzero(edges) / float(edges.size)) >= density_thresh

  def process_image(
      self, img_path, damage_conf=0.18, overlap_thresh=0.20, canny_thresh=0.03
  ):
    img = cv2.imread(img_path)
    if img is None:
      return None

    h, w = img.shape[:2]
    annotated = img.copy()
    overlay = img.copy()

    car_mask = self.get_car_mask(img)
    damage_res = self.m2.predict(source=img, conf=damage_conf, verbose=False)[0]

    if damage_res.boxes is not None:
      for i, box in enumerate(damage_res.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        box_area = (x2 - x1) * (y2 - y1)

        # 1. 車體重疊率過濾
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 1
        overlap = cv2.bitwise_and(roi_mask, roi_mask, mask=car_mask)
        if (
            np.count_nonzero(overlap) / float(box_area if box_area > 0 else 1)
        ) < overlap_thresh:
          continue

        # 2. Canny 紋理強度過濾
        if not self.check_canny_density(img[y1:y2, x1:x2], density_thresh=canny_thresh):
          continue

        # 繪製遮罩與標註
        if damage_res.masks is not None and i < len(damage_res.masks.data):
          m = cv2.resize(
              damage_res.masks.data[i].cpu().numpy().astype(np.uint8),
              (w, h),
              interpolation=cv2.INTER_NEAREST,
          )
          overlay[m == 1] = (255, 0, 150)

        cls_name = self.m2.names[int(box.cls[0])]
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"{cls_name} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
    return annotated

  def process_directory(
      self,
      input_dir,
      output_dir,
      damage_conf=0.18,
      overlap_thresh=0.20,
      canny_thresh=0.03,
  ):
    os.makedirs(output_dir, exist_ok=True)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    image_paths = [
        p for ext in exts for p in glob.glob(os.path.join(input_dir, ext))
    ]

    if not image_paths:
      print(f"Directory empty or no images found in: {input_dir}")
      return

    print(f"Processing {len(image_paths)} images...")
    for p in image_paths:
      res_img = self.process_image(
          p, damage_conf, overlap_thresh, canny_thresh
      )
      if res_img is not None:
        out_path = os.path.join(output_dir, f"result_{os.path.basename(p)}")
        cv2.imwrite(out_path, res_img)

    print(f"Processing complete. Saved to: {output_dir}")


if __name__ == "__main__":
  detector = CarDamageDetector(
      stage1_model="yolov8x-seg.pt", stage2_model="best.pt"
  )
  detector.process_directory(
      input_dir="damage_image",#可改成non_damage_image
      output_dir="damage_results",#可改成non_damage_result
      damage_conf=0.18,
      overlap_thresh=0.20,
      canny_thresh=0.03,
  )
  