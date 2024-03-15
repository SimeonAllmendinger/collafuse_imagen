from cleanfid import fid

for k in [0,200,400,600,800,1000]:
    score = fid.compute_fid(fdir1=f'/home/vault/btr0/btr0104h/collafuse/results/testing/{k}/generated', 
                            fdir2=f'/home/vault/btr0/btr0104h/collafuse/results/testing/{k}/real', 
                            mode="clean", 
                            model_name="clip_vit_b_32")

    print(score)