# Menggunakan base image Python 3.9 versi slim untuk ukuran lebih kecil
FROM python:3.9-slim

# Mengatur direktori kerja dalam container
WORKDIR /ml-backend

# Menyalin file requirements.txt ke dalam container
COPY requirements.txt .

# Menginstal dependencies dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh proyek ke dalam container
COPY . .

# Menyediakan port (opsional jika menggunakan web server)
EXPOSE 8081

# Menjalankan aplikasi (ubah 'app.py' dengan entry point Anda)
CMD ["python", "app.py"]
