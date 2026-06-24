# Face Anti-Spoofing Test Module

Test script đơn giản để kiểm tra khuôn mặt real hay fake.

## Cài đặt Dependencies

```bash
cd test_model
pip install -r requirements.txt
```

## Cách sử dụng

### 1. Chuẩn bị ảnh test
- Ảnh phải có tỉ lệ khung hình **3:4** (chiều rộng:chiều cao = 3:4)
- Ví dụ: 480x640, 600x800, 300x400
- Đặt ảnh vào folder `images/` hoặc bất kỳ đâu trên hệ thống

### 2. Chạy test

**Cách 1: Từ root directory**
```bash
python test_model/test_simple.py --image ./test_model/images/your_image.jpg
```

**Cách 2: Từ trong folder test_model**
```bash
cd test_model
python test_simple.py --image images/your_image.jpg
```

### 3. Các tùy chọn

```bash
python test_simple.py --image <image_path> \
                      --model_dir <path_to_models> \
                      --device_id <gpu_id>
```

| Tham số | Mô tả | Mặc định |
|---------|-------|---------|
| `--image` | Đường dẫn ảnh test | (bắt buộc) |
| `--model_dir` | Thư mục chứa models | `./resources/anti_spoof_models` |
| `--device_id` | GPU ID (0=GPU, CPU auto) | 0 |

## Ví dụ

```bash
# Test ảnh từ folder samples
python test_simple.py --image ./images/sample/face.jpg

# Test từ folder tùy chỉ
python test_simple.py --image D:/my_photos/test.jpg

# Sử dụng GPU 1
python test_simple.py --image ./test_model/images/face.jpg --device_id 1
```

## Kết quả

Script sẽ output:
- **REAL FACE**: Khuôn mặt thực (sống)
- **SPOOFING/FAKE**: Giả mạo (ảnh in, màn hình, mặt nạ, etc.)
- Độ tin cậy (confidence score) từ 0-1

## Cấu trúc folder đề xuất

```
test_model/
├── README.md
├── requirements.txt
├── test_simple.py
└── images/
    ├── test1.jpg
    ├── test2.jpg
    └── ...
```

## Lưu ý

- Ảnh phải có **khuôn mặt đầy đủ** trong frame
- Khuôn mặt không nên quay > 30 độ so với trục dọc
- Camera phải capture ảnh (không phải download từ web)
- Sử dụng PyTorch models từ `resources/anti_spoof_models`

## Troubleshooting

**Lỗi: "No face detected in image"**
- Đảm bảo khuôn mặt rõ ràng trong ảnh
- Thử ảnh khác

**Lỗi: "Image aspect ratio..."**
- Thay đổi kích thước ảnh để đạt tỉ lệ 3:4
- Script vẫn chạy được nhưng kết quả có thể không chính xác

**Lỗi: "CUDA out of memory"**
- Dùng `--device_id` với CPU (không xác định GPU)
- Hoặc giảm kích thước ảnh

## Liên kết

- GitHub: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- Models: `../resources/anti_spoof_models/`
