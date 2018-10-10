# SpatialNets


# step 1. python setup.py build
# step 2. pip install -e .
# step 3. python setup.py install

# Example

    from SpatialNets_pavia import SpatialNets 

    SpatialNets.convert_images(file_name = '/storage/geocloud/test/data/原始影像数据库/GF2/L1A/PMS/4m多光谱/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-MSS1.tiff', model_path = '/home/roo/tf-backup/mod-ZJ/DATA/train/SpatialNets_model_path.pth',save_file_name = '/storage/tmp/result.tif')