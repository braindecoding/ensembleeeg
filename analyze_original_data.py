#!/usr/bin/env python3
# analyze_original_data.py - Analisis mendalam data asli

import numpy as np

def analyze_original_data():
    """Analisis data asli dari file EP1.01.txt"""
    print("🔍 ANALISIS DATA ASLI EP1.01.txt")
    print("=" * 50)
    
    file_path = "Data/EP1.01.txt"
    
    # Analisis struktur file
    print("📊 Analisis Struktur File:")
    
    digit_6_samples = []
    digit_9_samples = []
    all_values = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= 10:  # Hanya analisis 10 baris pertama untuk sampel
                break
                
            if not line.strip():
                continue
                
            parts = line.split('\t')
            if len(parts) >= 7:
                try:
                    digit = int(parts[4])
                    channel = parts[3]
                    data_string = parts[6]
                    
                    # Parse nilai data
                    values = [float(x.strip()) for x in data_string.split(',') if x.strip()]
                    
                    print(f"  Baris {line_num+1}: Digit={digit}, Channel={channel}, Data points={len(values)}")
                    print(f"    Range nilai: {min(values):.2f} - {max(values):.2f}")
                    print(f"    Mean: {np.mean(values):.2f}, Std: {np.std(values):.2f}")
                    
                    all_values.extend(values)
                    
                    if digit == 6:
                        digit_6_samples.extend(values)
                    elif digit == 9:
                        digit_9_samples.extend(values)
                        
                except (ValueError, IndexError) as e:
                    print(f"  ⚠️ Error parsing line {line_num+1}: {e}")
    
    print(f"\n📈 Statistik Keseluruhan (dari 10 baris pertama):")
    print(f"  Total nilai: {len(all_values)}")
    print(f"  Range: {min(all_values):.2f} - {max(all_values):.2f}")
    print(f"  Mean: {np.mean(all_values):.2f}")
    print(f"  Std: {np.std(all_values):.2f}")
    
    if digit_6_samples and digit_9_samples:
        print(f"\n🔢 Perbandingan Digit 6 vs 9:")
        print(f"  Digit 6 - Mean: {np.mean(digit_6_samples):.2f}, Std: {np.std(digit_6_samples):.2f}")
        print(f"  Digit 9 - Mean: {np.mean(digit_9_samples):.2f}, Std: {np.std(digit_9_samples):.2f}")
        print(f"  Selisih mean: {abs(np.mean(digit_6_samples) - np.mean(digit_9_samples)):.2f}")
    
    # Analisis apakah ini data EEG yang wajar
    print(f"\n⚠️ EVALUASI KEASLIAN DATA:")
    
    mean_val = np.mean(all_values)
    if mean_val > 1000:
        print(f"  🚨 Range nilai terlalu tinggi untuk EEG (mean={mean_val:.0f})")
        print(f"      EEG normal: ±100 μV, data ini: ribuan")
    
    std_val = np.std(all_values)
    if std_val < 100:
        print(f"  🚨 Variabilitas terlalu rendah (std={std_val:.2f})")
        print(f"      EEG normal memiliki noise dan variasi yang lebih besar")
    
    # Cek apakah nilai terlalu 'smooth'
    sample_values = all_values[:1000]  # Ambil 1000 nilai pertama
    consecutive_diffs = [abs(sample_values[i+1] - sample_values[i]) for i in range(len(sample_values)-1)]
    mean_diff = np.mean(consecutive_diffs)
    
    print(f"\n📊 Analisis Smoothness:")
    print(f"  Mean perbedaan consecutive values: {mean_diff:.2f}")
    if mean_diff < 10:
        print(f"  🚨 Data terlalu smooth, kemungkinan sudah di-filter berlebihan")
    
    print(f"\n💡 KESIMPULAN:")
    print("Berdasarkan analisis data asli:")
    print("1. 📊 Data memiliki range nilai yang tidak wajar untuk EEG")
    print("2. 🔄 Kemungkinan data sudah melalui preprocessing yang ekstensif")
    print("3. ⚙️ Nilai dalam ribuan, bukan mikroVolt seperti EEG normal")
    print("4. 🎯 Ini menjelaskan mengapa model mudah mencapai confidence 1.0")

def check_data_distribution():
    """Cek distribusi data yang sudah diproses"""
    print(f"\n🔍 CEK DISTRIBUSI DATA TERPROSES")
    print("=" * 50)
    
    try:
        # Load data yang sudah diproses
        data = np.load("reshaped_data.npy")
        labels = np.load("labels.npy")
        
        print(f"📊 Data shape: {data.shape}")
        
        # Analisis per channel
        print(f"\n📈 Analisis per Channel:")
        for ch in range(min(5, data.shape[1])):  # Analisis 5 channel pertama
            ch_data = data[:, ch, :].flatten()
            print(f"  Channel {ch}: mean={ch_data.mean():.2f}, std={ch_data.std():.2f}, range={ch_data.min():.0f}-{ch_data.max():.0f}")
        
        # Analisis per kelas
        print(f"\n🔢 Analisis per Kelas:")
        digit6_data = data[labels == 0]
        digit9_data = data[labels == 1]
        
        print(f"  Digit 6: mean={digit6_data.mean():.2f}, std={digit6_data.std():.2f}")
        print(f"  Digit 9: mean={digit9_data.mean():.2f}, std={digit9_data.std():.2f}")
        print(f"  Selisih: {abs(digit6_data.mean() - digit9_data.mean()):.2f}")
        
        # Cek separabilitas
        print(f"\n🎯 Analisis Separabilitas:")
        overlap_ratio = abs(digit6_data.mean() - digit9_data.mean()) / (digit6_data.std() + digit9_data.std())
        print(f"  Overlap ratio: {overlap_ratio:.4f}")
        if overlap_ratio < 0.1:
            print(f"  🚨 Kelas terlalu mudah dipisahkan!")
        
    except FileNotFoundError:
        print("❌ File data terproses tidak ditemukan")

def main():
    """Main function"""
    analyze_original_data()
    check_data_distribution()

if __name__ == "__main__":
    main()
