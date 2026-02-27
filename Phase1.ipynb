# Phase 1: Automated Data Pipeline - Updated (Train/Val/Test)

## 2. Import Libraries
import os
import pandas as pd
import requests
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from pydub import AudioSegment
import warnings
import shutil
warnings.filterwarnings('ignore')
print("✓ All dependencies installed!")

print("✓ Libraries imported successfully!")
## 3. Upload CSV Files
from google.colab import files

print("Please upload BOTH CSV files:")
print("1. Training Dataset CSV (with nativity labels)")
print("2. Test Dataset CSV (unlabeled nativity)")
print("\nUpload now:")

uploaded = files.upload()

uploaded_files = list(uploaded.keys())
print(f"\n✓ Uploaded files: {uploaded_files}")

# Identify which is which
train_csv = None
test_csv = None

for file in uploaded_files:
    if 'training' in file.lower() or 'train' in file.lower():
        train_csv = file
    elif 'test' in file.lower():
        test_csv = file

# If auto-detection failed, ask user or assign based on order
if not train_csv or not test_csv:
    print("\n⚠️ Could not auto-detect files. Please verify:")
    if len(uploaded_files) >= 2:
        train_csv = uploaded_files[0]
        test_csv = uploaded_files[1]
        print(f"Assuming: {train_csv} = Training, {test_csv} = Test")

print(f"\nTraining CSV: {train_csv}")
print(f"Test CSV: {test_csv}")
## 4. Configuration & Setup
# Configuration
OUTPUT_DIR = "Phase1"
TARGET_SR = 16000  # 16kHz sampling rate
TRAIN_RATIO = 0.8  # 80% of training data
VAL_RATIO = 0.2    # 20% of training data

# Create output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# Create split directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

print(f"✓ Created directory structure in {OUTPUT_DIR}")
## 5. Load and Analyze Data
# Load both CSVs
df_train_full = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

print("=" * 80)
print("TRAINING DATASET (with labels)")
print("=" * 80)
print(f"Total records: {len(df_train_full)}")
print(f"\nLanguage distribution:")
print(df_train_full['language'].value_counts())
print(f"\nNativity distribution:")
print(df_train_full['nativity_status'].value_counts())

print("\n" + "=" * 80)
print("TEST DATASET (unlabeled - for final evaluation)")
print("=" * 80)
print(f"Total records: {len(df_test)}")
print(f"\nLanguage distribution:")
print(df_test['language'].value_counts())
print(f"\nNativity status: {df_test['nativity_status'].unique()} (unlabeled)")

print(f"\n" + "=" * 80)
print(f"TOTAL DATASET: {len(df_train_full) + len(df_test)} samples")
print("=" * 80)
## 6. Split Training Data into Train & Val
def split_train_val(df):
    """Split training dataset into train (80%) and val (20%) with stratification"""
    print("\nSplitting Training Dataset into Train/Val...\n")
    
    # Stratified split by language
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_RATIO,
        stratify=df['language'],
        random_state=42
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}% of training data)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}% of training data)")
    
    print("\nLanguage distribution per split:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df)]:
        print(f"\n{split_name}:")
        print(split_df['language'].value_counts())
    
    return train_df, val_df

train_df, val_df = split_train_val(df_train_full)

# Save metadata
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_metadata.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val_metadata.csv'), index=False)
df_test.to_csv(os.path.join(OUTPUT_DIR, 'test_metadata.csv'), index=False)

print("\n✓ Metadata saved for all splits")
## 7. Audio Processing Functions
def download_audio(url, save_path, max_retries=3):
    """Download audio file from URL"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                return False
    return False

def convert_to_16khz_mono_wav(input_path, output_path):
    """Convert audio to 16kHz Mono WAV with multiple strategies"""
    # Strategy 1: Try pydub (best for AAC, OGA, and various formats)
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(TARGET_SR)  # 16kHz
        audio.export(output_path, format='wav')
        if os.path.exists(input_path) and input_path != output_path:
            os.remove(input_path)
        return True
    except:
        pass
    
    # Strategy 2: Try librosa
    try:
        audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
        sf.write(output_path, audio, TARGET_SR)
        if os.path.exists(input_path) and input_path != output_path:
            os.remove(input_path)
        return True
    except:
        pass
    
    # Strategy 3: Try librosa with manual resampling
    try:
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sf.write(output_path, audio, TARGET_SR)
        if os.path.exists(input_path) and input_path != output_path:
            os.remove(input_path)
        return True
    except:
        pass
    
    return False

print("✓ Processing functions defined")
## 8. Process Dataset Splits
def process_dataset(df, split_name, has_labels=True):
    """Process a dataset split"""
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    successful = 0
    failed = 0
    failed_files = []
    
    print(f"\nProcessing {split_name.upper()} split ({len(df)} files)...")
    if not has_labels:
        print("  (Unlabeled test set - nativity status unknown)")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name}"):
        dp_id = row['dp_id']
        url = row['audio_url']
        language = row['language']
        
        # For test set, nativity is unknown ("-")
        if has_labels:
            nativity = row['nativity_status']
            filename = f"{dp_id}_{nativity}_{language}.wav"
        else:
            # Test set: no nativity label in filename
            filename = f"{dp_id}_{language}.wav"
        
        # Get file extension from URL
        file_ext = os.path.splitext(url)[1].lower()
        if file_ext not in ['.mp3', '.wav', '.aac', '.oga']:
            file_ext = '.mp3'
        
        temp_filename = f"{dp_id}_temp{file_ext}"
        temp_path = os.path.join(split_dir, temp_filename)
        final_path = os.path.join(split_dir, filename)
        
        # Download and convert
        if download_audio(url, temp_path):
            if convert_to_16khz_mono_wav(temp_path, final_path):
                successful += 1
            else:
                failed += 1
                failed_files.append({'dp_id': dp_id, 'url': url, 'split': split_name})
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            failed += 1
            failed_files.append({'dp_id': dp_id, 'url': url, 'split': split_name})
    
    print(f"✓ {split_name}: {successful} successful, {failed} failed")
    return successful, failed, failed_files

print("✓ Ready to process splits")
## 9. Run Complete Pipeline
print("=" * 80)
print("Starting Complete Pipeline Execution")
print("=" * 80)

total_successful = 0
total_failed = 0
all_failed_files = []

# Process train and val (labeled)
for split_name, split_df in [('train', train_df), ('val', val_df)]:
    successful, failed, failed_files = process_dataset(split_df, split_name, has_labels=True)
    total_successful += successful
    total_failed += failed
    all_failed_files.extend(failed_files)

# Process test (unlabeled)
successful, failed, failed_files = process_dataset(df_test, 'test', has_labels=False)
total_successful += successful
total_failed += failed
all_failed_files.extend(failed_files)

# Summary
print("\n" + "=" * 80)
print("Pipeline Execution Complete!")
print("=" * 80)
print(f"Total processed: {total_successful + total_failed}")
print(f"Successful: {total_successful}")
print(f"Failed: {total_failed}")
print(f"Success rate: {total_successful/(total_successful + total_failed)*100:.2f}%")

print(f"\nDataset Distribution:")
print(f"  Train: {len(train_df)} labeled samples (for training)")
print(f"  Val:   {len(val_df)} labeled samples (for validation)")
print(f"  Test:  {len(df_test)} UNLABELED samples (for final evaluation)")

if all_failed_files:
    print(f"\n⚠️  {len(all_failed_files)} files failed (likely corrupted at source)")
    failed_df = pd.DataFrame(all_failed_files)
    failed_df.to_csv(os.path.join(OUTPUT_DIR, 'failed_files.csv'), index=False)
    print("Failed files saved to: Phase1/failed_files.csv")
else:
    print("\n🎉 100% SUCCESS - All files processed!")

print("=" * 80)
## 10. Verify Output
# Count files in each split
for split in ['train', 'val', 'test']:
    split_path = os.path.join(OUTPUT_DIR, split)
    wav_files = [f for f in os.listdir(split_path) if f.endswith('.wav')]
    print(f"{split.upper()}: {len(wav_files)} WAV files")

# Show sample filenames
print("\nSample files from train (with nativity labels):")
train_files = [f for f in os.listdir(os.path.join(OUTPUT_DIR, 'train')) if f.endswith('.wav')][:5]
for f in train_files:
    print(f"  - {f}")

print("\nSample files from test (WITHOUT nativity labels):")
test_files = [f for f in os.listdir(os.path.join(OUTPUT_DIR, 'test')) if f.endswith('.wav')][:5]
for f in test_files:
    print(f"  - {f}")
## 11. Create ZIP for Download
# Create ZIP file
shutil.make_archive('Phase1', 'zip', OUTPUT_DIR)

# Get file size
zip_size = os.path.getsize('Phase1.zip') / (1024 * 1024)  # MB
print(f"✓ Created Phase1.zip ({zip_size:.2f} MB)")

# Download
from google.colab import files
print("\nDownloading Phase1.zip...")
files.download('Phase1.zip')
print("\n✅ Download complete! Extract the ZIP to get your train/val/test folders.")

