<pre><code>.
└── src/
    └── deepsc/
         ├── models/
         │      ├── deepsc/
         │      └── scbert/
         │              ├── scbert.py
         │              └── reversible.py   
         ├── data/
         │      ├── dataset.py
         │      ├── preprocess.py
         │      ├── download/
         │      │      ├── cellxgene/
         │      │      │      ├── build_index_list.py
         │      │      │      ├── data_config.py
         │      │      │      ├── download_partition.sh
         │      │      │      └── download_partition.py
         │      │      └── tripleca/
         │      │             ├── craw_3ca.py
         │      │             ├── download_3ca.py
         │      │             └── merge_and_filter_dataset.py
         │      └── preprocessing/  
         │             ├── config.py     
         │             ├── gene_name_normalization.py 
         │             ├── get_feature_name_3ca_cxg.py   
         │             ├── preprocess_3ca_merge.py   
         │             ├── preprocess_datasets_3ca.py   
         │             ├── preprocess_datasets_cellxgene.py  
         │             └── preprocess_datasets.py                   
         ├── pretrain/
         │      └── pretrain.py
         └── train/
                └── trainer.py  
</code></pre>