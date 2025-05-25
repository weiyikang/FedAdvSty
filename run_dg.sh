# pacs, MSCAD
nohup python main_dg.py --config PACS.yaml --target-domain art_painting -bp ../ --lr_adv 2.0 --con 0.0 --cdrm 0.0 --seed 2 --wandb 0 --gpu 3 > ./log/pacs_res18_adv_lr2_SupCon0_cdrm0_art_painting_seed2_avgcls.txt 2>&1 &

nohup python main_dg.py --config PACS.yaml --target-domain cartoon -bp ../ --lr_adv 2.0 --con 0.0 --cdrm 0.0 --seed 2 --wandb 0 --gpu 4 > ./log/pacs_res18_adv_lr2_SupCon0_cdrm0_cartoon_seed2_avgcls.txt 2>&1 &

nohup python main_dg.py --config PACS.yaml --target-domain photo -bp ../ --lr_adv 2.0 --con 0.0 --cdrm 0.0 --seed 2 --wandb 0 --gpu 5 > ./log/pacs_res18_adv_lr2_SupCon0_cdrm0_photo_seed2_avgcls.txt 2>&1 &

nohup python main_dg.py --config PACS.yaml --target-domain sketch -bp ../ --lr_adv 2.0 --con 0.0 --cdrm 0.0 --seed 2 --wandb 0 --gpu 6 > ./log/pacs_res18_adv_lr2_SupCon0_cdrm0_sketch_seed2_avgcls.txt 2>&1 &
