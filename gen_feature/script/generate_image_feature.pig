SET mapred.map.tasks.speculative.execution true;
SET mapred.reduce.tasks.speculative.execution true;
SET mapred.compress.map.output true;
SET default_parallel 100;
SET mapreduce.map.memory.mb 3071;
SET mapreduce.reduce.memory.mb 4092;
SET job.name '${job_name}';

sku_info = LOAD 'gdm.gdm_m03_item_sku_da' 
           USING org.apache.hcatalog.pig.HCatLoader();

sku_info = FILTER sku_info BY
                  dt == '${day}'                AND
                  item_second_cate_cd == '1345' AND
                  item_sku_id IS NOT NULL;

sku_info = FOREACH sku_info 
           GENERATE item_sku_id,
                    item_third_cate_cd; 

sku_attr = LOAD '${input_path}'
           USING PigStorage() AS (sku_id:chararray,
                                  cate_3:chararray,
                                  brand:chararray,
                                  attr:chararray);

info_mearge_join = JOIN sku_info BY item_sku_id,
                        sku_attr BY sku_id;

info_mearge = FOREACH info_mearge_join GENERATE
                  sku_info::item_sku_id           AS sku_id,
                  sku_info::item_third_cate_cd    AS third_cate_cd,
                  sku_attr::attr                  AS attr;

sku_image = LOAD 'ad_search.sku_main_image'
            USING org.apache.hcatalog.pig.HCatLoader();

sku_image = FOREACH sku_image GENERATE
                skuid  AS sku_id,
                image;


id_image_attr = JOIN info_mearge BY sku_id,
                     sku_image   BY sku_id;

id_image_attr = FOREACH id_image_attr GENERATE
                    info_mearge::sku_id           AS sku_id,
                    info_mearge::third_cate_cd    AS third_cate_cd,
                    sku_image::image              AS image,
                    info_mearge::attr             AS attr;

STORE id_image_attr INTO '${save_path}';
