function generate_pair_sku {
    INPUT_PATH=$1
    SAVE_PATH=$2

    hadoop fs -test -e $SAVE_PATH
    if [ $? -eq 0 ];then
         hadoop fs -rmr ${SAVE_PATH}
    fi

spark-submit \
    --master yarn-client \
    --queue root.bdp_jmart_ad.jd_ad_retr \
    --conf spark.akka.frameSize=150 \
    --conf spark.core.connection.ack.wait.timeout=6000 \
    --conf spark.rdd.compress=true \
    --conf spark.storage.memoryFraction=0.6 \
    --conf spark.driver.maxResultSize=20g \
    --num-executors 100 \
    --driver-memory 16g \
    --executor-memory 16g \
    --executor-cores 8 \
    script/generate_pair_sku.py ${INPUT_PATH} ${SAVE_PATH}
}
