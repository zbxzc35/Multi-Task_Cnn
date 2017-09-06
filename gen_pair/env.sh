HADOOP_PROJECT_HOME="/user/jd_ad/ads_reco/xuzhexuan/multi-task_cnn"

function log
{
    _ERR_HDR_FMT="[%s] %.23s %s[%s]: "
    _ERR_MSG_FMT="${_ERR_HDR_FMT}%s\n"
    local tag="UNKNOWN"
    local msg="WRONG PARAMETER"
    if [ $# -eq 2 ]; then
        tag=`echo $1 | tr [:lower:] [:upper:]`
        msg=$2
    fi
    printf "$_ERR_MSG_FMT" $tag $(date +%F.%T.%N) ${BASH_SOURCE[1]##*/} ${BASH_LINENO[0]} "$msg" | tee -a $LOG_FILE                                                     
}

if [[ -z $PROJECT_HOME ]]; then
    PROJECT_HOME="$(cd "`dirname "$0"`"/.; pwd)"     
fi

LOG_FILE="$PROJECT_HOME/log/log"
>$LOG_FILE
NOWTIME=`date "+%Y-%m-%d %H:%M:%S"`
echo ">>>>>>>>>>>>>>>>>>>>>>>>> $NOWTIME <<<<<<<<<<<<<<<<<<<<<<<<<" >>$LOG_FILE

LOG_FILE_DEBUG="$LOG_FILE.debug"
>$LOG_FILE_DEBUG
echo ">>>>>>>>>>>>>>>>>>>>>>>>> $NOWTIME <<<<<<<<<<<<<<<<<<<<<<<<<" >>$LOG_FILE_DEBUG
