import numpy as np

# Membagi data pelatihan menjadi subset pelatihan dan validasi
def split_train_validation_data(data, validation_ratio):
    n_samples = len(data)
    n_validation = int(n_samples * validation_ratio)

    validation_data = data[-n_validation:]
    train_data = data[:-n_validation]

    return train_data, validation_data

# Data pelatihan
train_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Contoh data pelatihan (harus diganti dengan data saham yang sebenarnya)
validation_ratio = 0.1  # Rasio validasi (10%)

# Membagi data pelatihan menjadi subset pelatihan dan validasi
train_data, validation_data = split_train_validation_data(train_data, validation_ratio)

# Menampilkan hasil pemisahan data
print("Data Pelatihan:", train_data)
print("Data Validasi:", validation_data)
