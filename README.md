# Laporan Proyek Machine Learning - Charles Wijaya

## Domain Proyek

Harga sebuah rumah sering kali diasosiasikan dengan hal-hal yang tampak jelas, seperti jumlah kamar tidur, luas bangunan, atau desain eksterior yang menarik. Namun, dalam kenyataannya, proses negosiasi harga rumah jauh lebih kompleks dan dipengaruhi oleh banyak variabel tersembunyi yang tidak selalu diperhatikan oleh pembeli awam. Misalnya, tinggi plafon di ruang bawah tanah, hingga jenis material yang digunakan untuk fondasi dapat memainkan peran penting dalam menentukan nilai akhir sebuah properti.

Proyek ini berfokus pada prediksi harga rumah berdasarkan dataset yang kaya akan fitur, yaitu data penjualan rumah di Ames, Iowa. Dataset ini terdiri dari 79 variabel penjelas yang menggambarkan hampir semua aspek fisik dan lingkungan dari rumah-rumah tersebut, mulai dari karakteristik bangunan, kondisi interior, hingga lokasi geografis. Dengan jumlah fitur sebanyak ini, proyek ini tidak hanya menantang secara teknis dalam pemodelan data, tetapi juga memberikan wawasan menarik tentang bagaimana faktor-faktor tersembunyi dapat memengaruhi harga rumah secara signifikan.

Masalah ini penting untuk diselesaikan karena mampu memberikan insight berharga baik bagi pembeli maupun penjual rumah. Model prediksi harga yang akurat dapat membantu pembeli menentukan apakah harga rumah wajar, dan di sisi lain, membantu penjual menetapkan harga yang kompetitif di pasar. Selain itu, pendekatan berbasis data juga membuka peluang untuk pengambilan keputusan yang lebih objektif di pasar properti yang sangat fluktuatif.
  
  Format Referensi: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) 

## Business Understanding

Harga rumah dipengaruhi oleh berbagai faktor yang kompleks dan tidak selalu dapat dipahami secara langsung oleh pembeli maupun penjual. Dengan adanya dataset yang kaya fitur seperti dalam kasus Ames, Iowa, proyek ini bertujuan untuk memanfaatkan teknik data science untuk mengungkap hubungan tersembunyi antara karakteristik rumah dan harga jual akhir.

Bagian laporan ini mencakup:

### Problem Statements

- Bagaimana hubungan antara karakteristik rumah (seperti ukuran, kondisi, dan lokasi) dengan harga jualnya?

- Bagaimana cara membangun model prediksi harga rumah yang akurat berdasarkan fitur-fitur tersebut?

- Fitur mana saja yang paling signifikan dalam memengaruhi harga rumah di Ames, Iowa?

### Goals

- Mengidentifikasi dan memahami korelasi antara berbagai fitur rumah dan harga jualnya sehingga dapat digunakan untuk proses prediksi.

- Mengembangkan model machine learning yang mampu memprediksi harga rumah secara akurat berdasarkan fitur yang tersedia.

- Menentukan fitur-fitur penting (feature importance) untuk memberikan insight yang berguna bagi pemangku kepentingan seperti agen properti, penjual, dan pembeli.
Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Menggunakan beberapa algoritma regresi populer seperti Neural Network, Random Forest Regressor, dan Gradient Boosting Regressor (XGBoost/LGBM) dan Lasso Model untuk membangun baseline model dan membandingkan performanya.
    - Mengukur performa model dengan metrik evaluasi yang relevan, yaitu RMSE
    - Melakukan feature engineering dan seleksi fitur, termasuk: Menangani missing value, Transformasi data (skewness, scaling, encoding), dan Menentukan fitur-fitur paling penting (feature importance)
    - Memberikan visualisasi eksploratif (EDA - Exploratory Data Analysis) untuk memahami distribusi data, hubungan antar fitur, dan mendeteksi outlier. Visual yang digunakan meliputi: Histogram dan boxplot untuk melihat distribusi, Scatter plot antara fitur signifikan dll.

## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview). Dataset ini berisi informasi rinci mengenai rumah-rumah di Ames, Iowa, yang mencakup 79 variabel penjelas (fitur) dan 1 target variabel, yaitu harga jual rumah (SalePrice). Data ini mencerminkan karakteristik fisik rumah, kondisi lingkungan, serta aspek bangunan yang dapat memengaruhi nilai pasar rumah tersebut.

Dataset terbagi menjadi dua bagian:

    train.csv — berisi data rumah dengan harga jual yang diketahui (1460 baris).

    test.csv — berisi data rumah tanpa harga jual (1459 baris), yang digunakan untuk prediksi akhir dalam kompetisi. 


### Variabel-variabel pada House Prices Kaggle dataset adalah sebagai berikut:

- SalePrice: Harga jual properti dalam dolar AS (variabel target)
- MSSubClass: Kelas bangunan
- MSZoning: Klasifikasi zona wilayah
- LotFrontage: Panjang (dalam kaki) properti yang berbatasan langsung dengan jalan
- LotArea: Luas tanah dalam satuan kaki persegi
- Street: Jenis akses jalan ke properti
- Alley: Jenis akses gang ke properti
- LotShape: Bentuk umum dari properti
- LandContour: Kontur (kerataan) tanah
- Utilities: Jenis utilitas (listrik, air, dll) yang tersedia
- LotConfig: Konfigurasi lahan
- LandSlope: Tingkat kemiringan lahan
- Neighborhood: Nama lingkungan di dalam batas kota Ames
- Condition1: Kedekatan dengan jalan utama atau rel kereta
- Condition2: Kedekatan dengan jalan utama atau rel kereta (jika ada yang kedua)
- BldgType: Jenis tempat tinggal
- HouseStyle: Gaya atau desain rumah
- OverallQual: Kualitas material dan penyelesaian keseluruhan
- OverallCond: Penilaian kondisi keseluruhan
- YearBuilt: Tahun rumah dibangun
- YearRemodAdd: Tahun renovasi atau penambahan bangunan
- RoofStyle: Jenis atap
- RoofMatl: Material atap
- Exterior1st: Material pelapis luar utama
- Exterior2nd: Material pelapis luar tambahan (jika ada)
- MasVnrType: Jenis pelapis batu bata (veneer)
- MasVnrArea: Luas pelapis batu bata (dalam kaki persegi)
- ExterQual: Kualitas material luar rumah
- ExterCond: Kondisi material luar rumah saat ini
- Foundation: Jenis pondasi
- BsmtQual: Tinggi basement (ruang bawah tanah)
- BsmtCond: Kondisi umum basement
- BsmtExposure: Jenis akses cahaya alami atau walkout pada basement
- BsmtFinType1: Kualitas area basement yang selesai dibangun (utama)
- BsmtFinSF1: Luas area basement yang selesai dibangun (utama)
- BsmtFinType2: Kualitas area basement yang selesai dibangun (sekunder)
- BsmtFinSF2: Luas area basement yang selesai dibangun (sekunder)
- BsmtUnfSF: Luas area basement yang belum selesai dibangun
- TotalBsmtSF: Total luas basement
- Heating: Jenis pemanas
- HeatingQC: Kualitas dan kondisi sistem pemanas
- CentralAir: Apakah terdapat pendingin udara terpusat
- Electrical: Sistem kelistrikan
- 1stFlrSF: Luas lantai pertama
- 2ndFlrSF: Luas lantai kedua
- LowQualFinSF: Luas bangunan jadi dengan kualitas rendah (semua lantai)
- GrLivArea: Luas area tinggal di atas permukaan tanah
- BsmtFullBath: Jumlah kamar mandi penuh di basement
- BsmtHalfBath: Jumlah kamar mandi setengah di basement
- FullBath: Jumlah kamar mandi penuh di atas tanah
- HalfBath: Jumlah kamar mandi setengah di atas tanah
- Bedroom: Jumlah kamar tidur di atas basement
- Kitchen: Jumlah dapur
- KitchenQual: Kualitas dapur
- TotRmsAbvGrd: Jumlah total ruangan di atas tanah (tidak termasuk kamar mandi)
- Functional: Penilaian fungsional rumah
- Fireplaces: Jumlah perapian
- FireplaceQu: Kualitas perapian
- GarageType: Lokasi garasi
- GarageYrBlt: Tahun garasi dibangun
- GarageFinish: Penyelesaian interior garasi
- GarageCars: Kapasitas garasi dalam jumlah mobil
- GarageArea: Luas garasi dalam kaki persegi
- GarageQual: Kualitas garasi
- GarageCond: Kondisi garasi
- PavedDrive: Apakah jalan masuk rumah sudah diaspal
- WoodDeckSF: Luas dek kayu dalam kaki persegi
- OpenPorchSF: Luas teras terbuka dalam kaki persegi
- EnclosedPorch: Luas teras tertutup dalam kaki persegi
- 3SsnPorch: Luas teras tiga musim dalam kaki persegi
- ScreenPorch: Luas teras berjaring dalam kaki persegi
- PoolArea: Luas kolam renang dalam kaki persegi
- PoolQC: Kualitas kolam renang
- Fence: Kualitas pagar
- MiscFeature: Fitur tambahan lainnya yang tidak termasuk kategori lain
- MiscVal: Nilai dari fitur tambahan
- MoSold: Bulan saat properti dijual
- YrSold: Tahun saat properti dijual
- SaleType: Jenis penjualan
- SaleCondition: Kondisi penjualan

**Rubrik/Kriteria Tambahan (Opsional)**:
Untuk memahami data secara lebih mendalam, saya melakukan beberapa teknik visualisasi, yaitu:

- Boxplot dibuat untuk mengidentifikasi nilai pencilan (outliers) dan distribusi setiap kolom numerik.
- Scatter plot digunakan untuk melihat hubungan antara fitur numerik dengan target (SalePrice).
- Histogram dan KDE plot membantu memahami distribusi data.

## Data Preparation

1. Handle NaN Value
2. Mengatasi Skewness
3. Remove Multicollinearity
4. Mengubah Categorical Menjadi Numeric
5. Train Test Split

**Rubrik/Kriteria Tambahan (Opsional)**: 
Handle NaN Value 

Penjelasan:
1. Melihat alasan mengapa ada NaN value misal seperti kasus Jumlah Fireplace QU sama dengan jumlah Fireplaces yang 0 artinya NAN karena tidak ada fireplaces kita jadikan kategori baru NA.

2. Mengganti dengan nilai median seperti kasus pada LotFrontage dan MasVnrArea.

3. Melakukan drop NaN value seperti kasus Eletrical ini kategori bisa menggunakan banyak metode bisa diambil yang paling sering muncul atau random dll. Akan tetapi NAN value hanya 1 baris saja sehingga saya drop aja.


Mengapa?
1. Membuat kategori baru 'NA' untuk fitur seperti FireplaceQu karena NaN-nya bukan kesalahan pencatatan, tetapi memang tidak ada fitur tersebut di rumah itu (contoh: tidak punya perapian). Jadi, 'NA' mencerminkan informasi penting bahwa rumah itu tidak memiliki fitur tersebut.

2. Median tidak terpengaruh oleh outliers dan distribusi fitur ini mungkin skewed, sehingga mean akan bias, sedangkan median lebih representatif.

3. Hanya terdapat 1 baris yang memiliki NaN karena tidak banyak memengaruhi dataset secara keseluruhan.


Mencoba memperbaiki kolom yang skew

Penjelasan

1. Menghitung skewness dari tiap kolom

2. Menentukan dan Melakukan transformasi logaritmik


Mengapa?
Jika fitur sangat skewed (misalnya skewness > 0.5 atau < -0.5), maka akan:

  - Mengurangi performa model.

  - Membuat model overfit atau kurang bisa generalisasi.


Transformasi logaritmik digunakan untuk:

  - Mengurangi skewness.

  - Membuat distribusi data lebih mendekati normal.

  - Menstabilkan variansi.


Remove Multicollinearity

Penjelasan

1. Disini saya mengecek multicollinearity menggunakan vif dimana diatas 10 udah terbukti ada multicollinearity

2. Setelah itu cek korelasi antar kolom

3. Saya ambil kolom yang memiliki korelasi dengan SalePrice lebih dari 0.1 atau lebih kecil dari -0.1


Mengapa?

Multicollinearity terjadi saat dua atau lebih fitur sangat berkorelasi satu sama lain.

Ini bisa membuat:

  - Model sulit mengestimasi koefisien secara akurat.

  - Interpretasi model jadi bias.

  - Meningkatkan variansi model.

Dengan menghitung VIF (Variance Inflation Factor), kita bisa tahu fitur mana yang redundan.

Dengan membuang fitur dengan VIF tinggi dan memilih fitur yang punya korelasi kuat dengan target (SalePrice), kita bisa:

  - Meningkatkan stabilitas dan interpretabilitas model.

  - Mengurangi overfitting.


Mengubah categorical menjadi numeric

Penjelasan

1. Melakukan One Hot Ecoding

Mengapa?

Model machine learning hanya bisa memproses angka, bukan teks sehingga harus kita ubah menjadi numeric. Karena tidak ada urutan logis maka digunakan One Hot Encoding.

One Hot Encoding:

  - Mencegah model membuat asumsi urutan/hierarki yang salah.

  - Menjaga independensi antar kategori.

  - Cocok untuk kategori yang tidak ordinal.


Train Test Split

Penjelasan

1. Membagi data menjadi set pelatihan dan set pengujian dengan ratio (80% train dan 20% test)

Mengapa?

Tujuan dari machine learning adalah membuat model yang bisa menggeneralisasi ke data baru.

Membagi data menjadi 80% training dan 20% testing membantu kita:

  - Melatih model dengan cukup data.

  - Mengukur performa pada data yang belum pernah dilihat model.

  - Menghindari overfitting karena kita tahu model tidak hanya menghafal data.


## Modeling
Model yang Digunakan:

Dalam tahapan ini, saya menggunakan tiga jenis model machine learning:

  - Neural Network (NN)

  - Random Forest Regressor

  - XGBoost Regressor

Setiap model memiliki peran dan keunggulannya masing-masing untuk menyelesaikan permasalahan regresi yaitu memprediksi SalePrice.

**Rubrik/Kriteria Tambahan (Opsional)**: 

Neural Network

Arsitektur dan Parameter:

    - 4 hidden layers:

        Ukuran neuron: 100, 250, 100, 50

        Aktivasi: ReLU

    - Dropout setelah setiap layer untuk mencegah overfitting

    - Output layer tanpa aktivasi karena ini adalah regresi

    - Optimizer: Adam dengan learning rate 0.0001

    - Loss function: Mean Squared Error (MSE)

Kelebihan:

    - Mampu menangkap hubungan kompleks dan non-linear.

    - Fleksibel dan dapat disesuaikan dengan data.

Kekurangan:

  - Butuh tuning yang lebih rumit.

  - Lebih sensitif terhadap overfitting jika data tidak cukup besar.

  - Waktu pelatihan lebih lama.

Random Forest Regressor

Parameter:

  - n_estimators=100

  - random_state=0 untuk reprodusibilitas

  - oob_score=True untuk validasi internal

Kelebihan:

  - Robust terhadap overfitting.

  - Tidak perlu scaling data.

  - Dapat menangani missing value dan data kategori dengan baik.

Kekurangan:

  - Kurang baik untuk data sangat besar karena waktu pelatihan bisa lama.

  - Hasil sulit untuk diinterpretasi (black box model).

XGBoost Regressor
Parameter:

  - learning_rate=0.01: kecil agar model belajar perlahan dan stabil

  - n_estimators=3460: banyak estimators agar model lebih kuat

  - max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.7

  - reg_alpha=0.00006: regularisasi L1 untuk mencegah overfitting

  - objective='reg:squarederror': digunakan untuk regresi

Kelebihan:

  - Sangat akurat dan efisien.

  - Memiliki kontrol overfitting yang baik melalui parameter regularisasi.

  - Salah satu algoritma paling populer untuk kompetisi seperti Kaggle.

Kekurangan:

  - Butuh banyak hyperparameter tuning agar optimal.

  - Waktu pelatihan bisa panjang.


Untuk mendapatkan solusi terbaik maka saya akan melakukan penggabungan beberapa model dimana rationya berdasarkan seberapa baik model tersebut:

predictions_ensemble = predictions_nn.reshape(1,-1) * 0.2 + predictions_random_forest.reshape(1,-1) * 0.1 + predictions_xgboost.reshape(1,-1) * 0.5 + y_pred_best.reshape(1,-1) * 0.2


## Evaluation

Karena permasalahan ini merupakan regresi (memprediksi nilai kontinu yaitu SalePrice), maka metrik evaluasi yang sesuai adalah RMSE.


**Rubrik/Kriteria Tambahan (Opsional)**: 

RMSE (Root Mean Squared Error) mengukur rata-rata kesalahan antara nilai prediksi dan nilai aktual dalam satuan yang sama dengan target (SalePrice dalam hal ini).

Formula RMSE:

![Formula RMSE](/rumus_rmse.png)
