# customer-prediction
Developed and deployed an XGBoost classification model using FastAPI to predict customer campaign response, integrating a full preprocessing pipeline and achieving 89% accuracy on test data.

Proyek ini adalah implementasi *Machine Learning* untuk memprediksi apakah seorang nasabah bank akan berlangganan deposito berjangka (*term deposit*) berdasarkan data kampanye pemasaran (*telemarketing*). 

Model machine learning dikembangkan menggunakan algoritma **XGBoost** dan dioptimasi dengan **SMOTE** untuk menangani masalah ketidakseimbangan kelas (*imbalanced data*). Model yang telah dilatih kemudian di-deploy sebagai web API menggunakan **FastAPI**.

## 📁 Struktur Repositori

* **`uas-model-deployment.ipynb`**: Jupyter Notebook yang berisi seluruh proses eksplorasi data (EDA), pra-pemrosesan data, pelatihan model dengan XGBoost dan SMOTE/ADASYN, evaluasi model, serta *export* model menjadi file *pickle*.
* **`prediction.py`**: Script utama FastAPI yang memuat model dan berbagai *encoder*, serta menyediakan *endpoint* API untuk menerima data input dan mengembalikan hasil prediksi.
* **`XGBnSMOTE.pkl`**: File model klasifikasi XGBoost yang telah dilatih secara optimal.
* **`one_hot.pkl`**: File *One-Hot Encoder* untuk memproses fitur kategorikal nominal (`job`, `marital`, `education`, `contact`, `poutcome`).
* **`binary.pkl`**: File *Label Encoder* untuk memproses fitur biner (`housing`, `loan`).
* **`ordinal.pkl`**: File *Ordinal Encoder* untuk memproses fitur yang memiliki urutan/tingkatan (`month`, `day_of_week`).
* **`data_1D.csv`**: Dataset Bank Marketing mentah yang digunakan untuk melatih dan menguji model.

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3.13
* **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
* **Data Manipulasi:** Pandas, NumPy
* **API Deployment:** FastAPI, Uvicorn
