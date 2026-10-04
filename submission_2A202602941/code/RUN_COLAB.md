# Chạy trên Google Colab

1. Upload các thư mục `data`, `scripts`, `templates`, `submission_2A202602941`
   của dự án vào `MyDrive/Labs/K4-Track4-Day1-Neuralnetwork`.
   Không upload `.venv`, `.git`, `.verification` hoặc `__pycache__`.
2. Mở `submission_2A202602941/code/lab.ipynb` trên Colab.
3. Chọn Runtime → Change runtime type → GPU (T4 nếu có).
4. Cell đầu có `COLAB_REPO`: sửa nếu bạn upload vào vị trí khác.
5. Chạy Run all; chấp nhận mount Google Drive khi Colab yêu cầu.

Mặc định notebook chạy 20 epoch, 3 seed baseline, tìm lr và thí nghiệm 7 chủ đề.
Mỗi run lưu ngay JSON và PNG. Cuối cùng xuất dự đoán eval, chạy script chấm,
vẽ ma trận nhầm lẫn và tạo `experiments.xlsx` theo template.

Để kiểm tra nhanh trước, thay dòng `SMOKE = ...` trong cell đầu thành
`SMOKE = True`. Kết quả kiểm tra nằm trong `submission_2A202602941/_smoke`;
không dùng để nộp hoặc viết báo cáo. Đổi về `SMOKE = False` cho lượt chạy đầy đủ.

Notebook cần các module Python cùng dữ liệu/scripts/template; chỉ mở notebook
từ GitHub không tự cung cấp những file này. Không ghi đè code hoàn chỉnh bằng
bản khung cũ. Dữ liệu processed được tạo tự động khi chưa có.

Lưu notebook có output vào `submission_2A202602941/code/lab.ipynb`. Mở Excel
hoặc LibreOffice để tính lại công thức bảng. Viết `REPORT.md` theo template,
giải thích bằng số liệu chạy đầy đủ, không dùng các số của smoke test.

Chọn cấu hình bằng validation trước khi chấm eval. Chạy lại Part 2–3 vì điểm eval
chưa tốt rồi chọn cấu hình mới là dùng eval để tối ưu, không đúng quy định lab.

## Chạy một cấu hình từ terminal

Từ thư mục gốc dự án:

```bash
python scripts/split_data.py
python submission_2A202602941/code/train.py --lr 0.03 --epochs 20 --exp-id base-s1
```

CLI này chỉ train/validation và lưu JSON. Notebook thực hiện toàn bộ lab.
