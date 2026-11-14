# Pen Grip Recognition

Dự án nhận diện 9 kiểu cầm bút từ video bằng MediaPipe Tasks Hand Landmarker và MLP scikit-learn.

## MediaPipe Hand Landmarker task model

Các script dùng MediaPipe Tasks `HandLandmarker`. Tải model bundle `hand_landmarker.task` từ [Google MediaPipe Models](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) và đặt tại `models/hand_landmarker.task` trước khi preview, tạo dataset hoặc chạy webcam.

## Cấu trúc dữ liệu

Đặt video vào thư mục theo nhãn trong `data/raw_videos/`. Tên file theo mẫu `User01_Right_01.mov` hoặc `User02_Left_01.mp4`; phần đầu là mã người dùng và dùng để chia train/test theo người.

- `0_Dynamic_Tripod`
- `1_Thumb_Wrap`
- `2_Thumb_Tuck`
- `3_Thumb_Over_Middle`
- `4_Fisted_Grip`
- `5_Grip_Too_Low`
- `6_Grip_Too_High`
- `7_Hooked_Wrist`
- `8_No_Pen`

Quay rõ bàn tay và bút, đủ sáng, giữ camera cố định. Mỗi nhãn nên có nhiều video từ nhiều người. Với `4_No_Pen`, cần quay bàn tay không cầm bút. Các nhãn khác cần thể hiện nhất quán kiểu cầm. Giữ nguyên nhiều mẫu theo các góc nhìn tự nhiên.

## Cài đặt và chạy

```powershell
cd D:\pen-grip-recognition
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/1_preview_landmarks.py data/raw_videos/0_Dynamic_Tripod/User01_Right_01.mov
python scripts/2_build_dataset.py
python scripts/3_train_model.py
python scripts/4_realtime_webcam.py
```

`2_build_dataset.py` và `4_realtime_webcam.py` cùng gọi `src.features.extract_features()` để tiền xử lý nhất quán. Hàm ghi 69 đặc trưng: 63 tọa độ landmark đã chuẩn hóa; 2 cờ handedness; 2 khoảng cách đầu ngón tay; và sin/cos hướng cổ tay trước khi xoay chuẩn hóa. Chỉ dữ liệu từ tay trái được đưa vào dataset. CSV lưu `user_id` (tách từ tên file, ví dụ `u1_2.mp4`) và `label` (0–8). Khi huấn luyện, `GroupShuffleSplit` chia theo user để đánh giá khả năng tổng quát hóa sang người mới. Webcam yêu cầu confidence ít nhất 0.70 và class `TemporalSmoother` dùng cửa sổ 15 mẫu, tỷ lệ phiếu trên 75%, cùng cảnh báo tức thời khi No Pen vượt ngưỡng.

Model và ma trận nhầm lẫn được ghi trong `models/`. Tệp ONNX được xuất nếu `skl2onnx` tương thích với các phiên bản thư viện đã cài. Cấu hình ngưỡng tin cậy và cửa sổ làm mượt nằm trong `src/config.py`.
