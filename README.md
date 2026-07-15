# Are Counterfactual HR Intervention Scenarios Feasible?

## A Pilot Audit of Employees with High Predicted Attrition Risk

Pipeline nghiên cứu đánh giá liệu các kịch bản can thiệp HR có thể làm **attrition score** của mô hình giảm xuống dưới ngưỡng high-risk hay không trên bộ IBM HR Analytics Employee Attrition & Performance.

## Tài liệu chính

- Paper hoàn chỉnh: [`Paper.pdf`](Paper.pdf)
- Mã nguồn LaTeX của paper: [`main.tex`](main.tex)
- Research novelty statement: [`CoverLetter.pdf`](CoverLetter.pdf)
- Slide thuyết trình: [`slides/hr-counterfactual-feasibility-audit.pptx`](slides/hr-counterfactual-feasibility-audit.pptx)

Dataset gốc và các artifact sinh tự động không được lưu trong repository. Chúng được tạo lại theo hướng dẫn bên dưới và được loại khỏi Git bằng `.gitignore`.

## Cài đặt và chạy

Các artifact nộp kèm đã được kiểm tra bằng **Python 3.14.3** và bộ phiên bản khóa trong `requirements-lock.txt`. Để tái tạo đúng môi trường đã kiểm tra:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-lock.txt
python3 run_pipeline.py
```

`requirements.txt` giữ khoảng phiên bản rộng hơn cho môi trường phát triển và có thêm Jupyter để chạy notebook. Kết quả có tính ngẫu nhiên của DiCE/XGBoost có thể khác giữa phiên bản hoặc nền tảng; khi đối chiếu artifact nên ưu tiên file lock.

## Chuẩn bị dữ liệu

Raw CSV không được phân phối cùng repository. Tải bộ [IBM HR Analytics Employee Attrition & Performance trên Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), sau đó đặt file tại:

```text
data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Kiểm tra đúng phiên bản dữ liệu đã dùng để tạo paper:

```bash
shasum -a 256 data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

SHA-256 kỳ vọng:

```text
a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659
```

Trên macOS, XGBoost có thể cần OpenMP:

```bash
brew install libomp
```

Nếu máy đã có `libomp.dylib` đi kèm scikit-learn nhưng không có Homebrew, có thể chạy:

```bash
DYLD_LIBRARY_PATH="$(python3 -c 'import pathlib, sklearn; print(pathlib.Path(sklearn.__file__).parent / ".dylibs")')" python3 run_pipeline.py
```

Các giả định nghiên cứu có thể thay đổi cho sensitivity nằm trong `config.yaml`; hyperparameter cố định của ba candidate model được ghi trong `src/hr_recourse/data_modeling.py` và trong paper. Để chạy nhanh khi phát triển, có thể tắt riêng SHAP, DiCE, stability, sensitivity hoặc figure trong nhóm `run`; cấu hình mặc định bật đầy đủ các phân tích đã cam kết.

Khi tắt DiCE, Level 1/2 được ghi là `skipped`. Employee không có Level 3 feasible scenario sẽ nhận `analysis_status=partial` thay vì bị kết luận sai là không có model recourse. Artifact thuộc phase bị tắt được xóa để tránh trộn với kết quả của run trước.

Chạy test:

```bash
python3 -m pytest -q
```

Build paper (sau khi cài MacTeX/BasicTeX):

```bash
latexmk -pdf main.tex
```

Paper sử dụng trực tiếp các hình trong `outputs/figures`, vì vậy nên chạy pipeline trước khi build LaTeX.

Slide thuyết trình được lưu tại [`slides/hr-counterfactual-feasibility-audit.pptx`](slides/hr-counterfactual-feasibility-audit.pptx).

## Pipeline

1. Kiểm tra schema/chất lượng dữ liệu và chia stratified 60/20/20.
2. Fit Logistic Regression, Random Forest và XGBoost bằng preprocessing chỉ học từ train.
3. Chọn model theo validation Average Precision và chọn threshold tối đa F2 trên validation.
4. Báo cáo test metrics và xác định test employees có `score >= threshold`.
5. Tính SHAP theo feature gốc và năm nhóm biến HR.
6. Sinh DiCE Level 1/2 với threshold adapter, sau đó chấm lại bằng attrition score gốc.
7. Exhaustive grid Level 3 cho `MonthlyIncome`, `StockOptionLevel`, `OverTime`.
8. Audit feasibility, rank scenario và phân loại từng employee.
9. Chạy stability theo seed và one-factor-at-a-time sensitivity.

Đầu ra chính:

```text
outputs/tables/hr_intervention_feasibility_assessment.csv
```

Các model, scenario, bảng phụ và hình được lưu lần lượt trong `outputs/models`, `outputs/scenarios`, `outputs/tables` và `outputs/figures`. Mỗi scenario lưu `changes_json` với giá trị trước/sau của từng biến thay đổi, cho phép dựng lại hồ sơ và chấm điểm độc lập.

## Diễn giải đúng phạm vi

- Dataset IBM là dữ liệu **synthetic**, phù hợp cho minh họa phương pháp nhưng không đại diện cho một tổ chức thực tế.
- `predict_proba` được gọi là **attrition score**; pipeline không khẳng định đây là xác suất đã calibration.
- Counterfactual và SHAP mô tả logic của mô hình, không chứng minh quan hệ nhân quả.
- Grid tăng lương 5%–30% là không gian nghiên cứu, không phải chính sách hay đề xuất tăng lương thực tế.
- Output không được dùng để tự động hóa quyết định tuyển dụng, kỷ luật, sa thải, thăng chức hay đãi ngộ. Mọi ứng dụng thực tế cần human review, kiểm tra fairness, privacy, legal compliance và dữ liệu phù hợp bối cảnh.
