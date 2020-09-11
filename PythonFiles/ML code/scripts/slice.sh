
#ROOT=/raid/data/home/hongzhi/USC/3D
ROOT=.
IMAGE_ROOT="/home/jpfrancis/Development/transfer/redo/"

#for group in test train valid; do 
for group in redo; do
  #IDs=`cat $ROOT/../2D/mixlabel/${group}_IDs`
  IDs=`cat $ROOT/${group}_IDs`
  for id in $IDs; do
    '''
    oimg=$IMAGE_ROOT/${id}.nii.gz
    echo $oimg

    if [ ! -d $IMAGE_ROOT/$id/ ]; then 
        mkdir $IMAGE_ROOT/$id/
    fi
    for a in x z; do 
        if [ ! -d $IMAGE_ROOT/$id/$a ]; then
            mkdir $IMAGE_ROOT/$id/$a
        fi

        nimg=$IMAGE_ROOT/$id/$a/${id}_2D_${a}_%0d.nii.gz
        ~/Development/c3d $oimg -slice $a 0:-1 -oo $nimg
    done
    '''
    oimg=$IMAGE_ROOT/${id}_labels.nii.gz
    echo $oimg

    if [ ! -d $IMAGE_ROOT/Label_3D/$id/ ]; then
        mkdir $IMAGE_ROOT/Label_3D/$id/
    fi
    for a in x z; do
        if [ ! -d $IMAGE_ROOT/Label_3D/$id/$a ]; then
            mkdir $IMAGE_ROOT/Label_3D/$id/$a
        fi

        nimg=$IMAGE_ROOT/Label_3D/$id/$a/${id}_labels_2D_${a}_%0d.nii.gz
        ~/Development/c3d $oimg -slice $a 0:-1 -oo $nimg
    done
  done
done

'''
IDs=`cat $ROOT/unlabeled_IDs`

echo $IDs
for id in $IDs; do
    oimg=$ROOT/Unlabeled_3D/$id.nii.gz
    echo $oimg

    if [ ! -d $ROOT/Unlabeled_3D/$id/ ]; then
        mkdir $ROOT/Unlabeled_3D/$id/
    fi
    for a in x y z; do
        if [ ! -d $ROOT/Unlabeled_3D/$id/$a ]; then
            mkdir $ROOT/Unlabeled_3D/$id/$a
        fi

        nimg=$ROOT/Unlabeled_3D/$id/$a/${id}_2D_${a}_%0d.nii.gz
        ~/c3d $oimg -slice $a 0:-1 -oo $nimg
    done
done
'''

