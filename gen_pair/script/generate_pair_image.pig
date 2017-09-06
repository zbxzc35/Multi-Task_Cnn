SET mapred.map.tasks.speculative.execution true;
SET mapred.reduce.tasks.speculative.execution true;
SET mapred.compress.map.output true;
SET default_parallel 100;
SET mapreduce.map.memory.mb 3071;
SET mapreduce.reduce.memory.mb 4092;
SET job.name '${job_name}';

pair_sku = LOAD '${input_path}'
               USING PigStorage() AS (cid3:chararray,
                                      sku1:chararray,
                                      attrs1:chararray,
                                      sku2:chararray,
                                      attrs2:chararray);

sku_inf = LOAD '/user/jd_ad/ads_reco/wangjincheng/sku_text_info'
           USING PigStorage() AS (sku_id:chararray,
                                  sku_name:chararray,
                                  item_name:chararray,
                                  item_desc:chararray,
                                  barndname_full:chararray,
                                  cid1:chararray,
                                  cid2:chararray,
                                  cid3:chararray,
                                  slogan:chararray,
                                  item_type:chararray,
                                  title:chararray,
                                  query:chararray,
                                  image:chararray,
                                  sku_attrs:chararray);

sku_inf = FILTER sku_inf BY cid1 IN ('鞋靴','运动户外','服饰内衣','日用百货');
sku_inf = FOREACH sku_inf GENERATE sku_id, image;

pair_sku_image1 = JOIN pair_sku BY sku1 LEFT OUTER, sku_inf BY sku_id;
pair_sku_image1 = FOREACH pair_sku_image1 GENERATE
                    pair_sku::cid3 AS cid3,
                    pair_sku::sku1 AS sku1,
                    sku_inf::image AS image1,
                    pair_sku::attrs1 AS attrs1,
                    pair_sku::sku2 AS sku2,
                    pair_sku::attrs2 AS attrs2;

pair_sku_image2 = JOIN pair_sku_image1 BY sku2 LEFT OUTER, sku_inf BY sku_id;
pair_sku_image = FOREACH pair_sku_image2 GENERATE
                    pair_sku_image1::cid3 AS cid3,
                    pair_sku_image1::sku1 AS sku1,
                    pair_sku_image1::image1 AS image1,
                    pair_sku_image1::attrs1 AS attrs1,
                    pair_sku_image1::sku2 AS sku2,
                    sku_inf::image AS image2,
                    pair_sku_image1::attrs2 AS attrs2;

STORE pair_sku_image INTO '${save_path}';


