```
# Create a virtual environment using either venv or conda. For example, 
python3 -m venv constricon_venv
source constricon_venv/bin/activate

# Install and use the software
pip install unigradicon
git clone https://github.com/uncbiag/mouse_brain_translucence

cd mouse_brain_translucence
python cli.py --fixed your_data_path/000_auto_resampled_12.tif --moving your_data_path/001_auto_resampled_12.tif --transform_out f_000_m_001.hdf5
```
