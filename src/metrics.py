#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

TAG_FLOAT = 202021.25  # Tag de Sintel pour les .flo

###############################################################################
# 1) Lecture d’un .flo
###############################################################################

def read_flo_file(filename):
    """
    Lit un fichier .flo (format MPI-Sintel) et renvoie un numpy array (H,W,2).
    """
    with open(filename, 'rb') as f:
        tag = np.frombuffer(f.read(4), dtype=np.float32)[0]
        if tag != TAG_FLOAT:
            raise ValueError(f"Fichier %s invalide (tag=%s)" % (filename, tag))
        w = np.frombuffer(f.read(4), dtype=np.int32)[0]
        h = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(4*h*w*2), dtype=np.float32)
        flow = np.reshape(data, (h, w, 2))
    return flow


###############################################################################
# 2) Calcul EPE / AAE
###############################################################################

def compute_epe_and_aae(flow_pred, flow_gt):
    """
    Calcule l'End-Point Error (EPE) et l'Angular Error (AAE) en degrés
    entre flow_pred et flow_gt (chacun shape (H, W, 2)).
    Retourne (epe_moy, aae_degrees_moy).
    """
    diff = flow_pred - flow_gt
    epe_map = np.sqrt(diff[:,:,0]**2 + diff[:,:,1]**2)
    epe = np.mean(epe_map)

    # AAE
    u_p, v_p = flow_pred[:,:,0], flow_pred[:,:,1]
    u_g, v_g = flow_gt[:,:,0], flow_gt[:,:,1]
    mag_p = np.sqrt(u_p**2 + v_p**2)
    mag_g = np.sqrt(u_g**2 + v_g**2)
    dot   = u_p*u_g + v_p*v_g
    eps   = 1e-9
    cos   = dot / (mag_p * mag_g + eps)
    cos   = np.clip(cos, -1.0, 1.0)
    angle_map = np.arccos(cos)  # radians
    aae_deg   = np.mean(angle_map) * (180.0 / math.pi)

    return epe, aae_deg


###############################################################################
# 3) Warp + MSE (pour GITW)
###############################################################################

def flow_warp(img1, flow):
    """
    Remappe img1 à l’aide de flow (H,W,2).
    flow[y,x] = (u, v).
    => (y, x) va chercher la couleur en (y - v, x - u) (warp inverse).
    """
    h, w = flow.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            u, v = flow[y,x]
            map_x[y, x] = x - u
            map_y[y, x] = y - v

    warped = cv2.remap(
        img1, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped

def compute_mse(imgA, imgB):
    """
    Calcule la MSE entre imgA et imgB (même shape (H,W,3)).
    """
    diff = imgA.astype(np.float32) - imgB.astype(np.float32)
    mse = np.mean(diff**2)
    return mse


###############################################################################
# 4) Évaluation sur MPI-Sintel + Graphiques
###############################################################################

def evaluate_sintel(sintel_base):
    """
    Évalue EPE + AAE sur MPI-Sintel
    en comparant flow_pred_raft/<clean|final>/<seq>/*.flo
    avec flow/<seq>/*.flo (ground truth).
    
    Génère des courbes :
      - EPE / AAE en fonction de l'index de frame (pour chaque séquence)
      - Moyenne EPE / AAE par séquence (bar plot)
    """
    sequences = ['alley_2', 'market_2', 'temple_3']
    subfolders_pred = ['clean', 'final']

    # On va stocker les courbes d'EPE/AAE pour chaque sous-folder et séquence
    # ex: data_sintel['clean']['alley_2'] = dict(epe_list=[...], aae_list=[...])
    data_sintel = {
        sf: {seq: {'epe_list': [], 'aae_list': []} for seq in sequences}
        for sf in subfolders_pred
    }

    for subf in subfolders_pred:
        for seq_name in sequences:
            gt_dir   = os.path.join(sintel_base, 'training', 'flow', seq_name)
            pred_dir = os.path.join(sintel_base, 'training', 'flow_pred_raft', subf, seq_name)
            if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
                print(f"[WARN] GT or pred dir missing for {subf}/{seq_name}")
                continue

            # Lister .flo ground truth
            gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.flo')])
            for idx, f_gt in enumerate(gt_files):
                path_gt   = os.path.join(gt_dir, f_gt)
                path_pred = os.path.join(pred_dir, f_gt)
                if not os.path.isfile(path_pred):
                    continue

                flow_gt   = read_flo_file(path_gt)
                flow_pred = read_flo_file(path_pred)

                epe, aae = compute_epe_and_aae(flow_pred, flow_gt)
                data_sintel[subf][seq_name]['epe_list'].append(epe)
                data_sintel[subf][seq_name]['aae_list'].append(aae)

    # 4.1) Graphique EPE/AAE par frame (index) dans chaque séquence
    for subf in subfolders_pred:
        for seq_name in sequences:
            epe_list = data_sintel[subf][seq_name]['epe_list']
            aae_list = data_sintel[subf][seq_name]['aae_list']
            if not epe_list:
                continue

            indexes = range(len(epe_list))
            
            plt.figure(figsize=(8,5))
            plt.plot(indexes, epe_list, label='EPE')
            plt.plot(indexes, aae_list, label='AAE')
            plt.title(f"Sintel {subf}/{seq_name} - EPE & AAE vs. frame index")
            plt.xlabel("Frame index")
            plt.ylabel("Metric value")
            plt.legend()
            plt.grid(True)
            plt.show()
            # plt.savefig(f"sintel_{subf}_{seq_name}_metrics.png")  # pour sauver en fichier
            
    # 4.2) Graphique de l’EPE moyen + AAE moyen par séquence (bar plot)
    # On veut : X-axis = [alley_2, market_2, temple_3], Y = EPE moyen (barres)
    # On peut faire 2 barres (clean/final) côte à côte, et idem pour AAE.
    # => Ou on fait un graphe EPE, un graphe AAE.
    
    for metric_name in ['epe_list', 'aae_list']:
        plt.figure(figsize=(6,5))
        
        # Barres groupées : subfolders_pred sur l'axe X *et* sequences
        # Simplifions : x-axis = sequences, pour chaque sequence on a 2 barres (clean/final).
        x_positions = np.arange(len(sequences))
        bar_width = 0.3
        
        # Calcule la moyenne par (sf, seq)
        means_clean = []
        means_final = []
        for seq_name in sequences:
            epe_clean = data_sintel['clean'][seq_name][metric_name]
            epe_final = data_sintel['final'][seq_name][metric_name]
            mean_clean = np.mean(epe_clean) if epe_clean else None
            mean_final = np.mean(epe_final) if epe_final else None
            means_clean.append(mean_clean if mean_clean is not None else 0)
            means_final.append(mean_final if mean_final is not None else 0)

        # On trace 2 barres
        plt.bar(x_positions - bar_width/2, means_clean, width=bar_width, label='clean')
        plt.bar(x_positions + bar_width/2, means_final, width=bar_width, label='final')
        
        # Label + ticks
        plt.xticks(x_positions, sequences)
        plt.ylabel(metric_name.replace('_list','').upper())  # "EPE" ou "AAE"
        plt.title(f"Mean {metric_name.replace('_list','').upper()} by sequence (Sintel)")
        plt.legend()
        plt.grid(axis='y')
        plt.show()
        # plt.savefig(f"sintel_{metric_name}_bar.png")


###############################################################################
# 5) Évaluation sur GITW + Graphiques
###############################################################################

def evaluate_gitw(gitw_base):
    """
    Calcule MSE pour GITW en warpant img1 -> img2, 
    et produit des courbes:
      - MSE par frame
      - MSE moyen par séquence (bar plot)
    """
    objects = ['Bowl', 'CanOfCocaCola', 'Rice']

    # data_gitw[obj][subd] = dict(mse_list=[])
    data_gitw = {}

    compteur = 0
    total = len(objects)

    for obj in objects:
        compteur += 1
        print(f"Processing {obj} ({compteur}/{total})")
        obj_path = os.path.join(gitw_base, obj)
        if not os.path.isdir(obj_path):
            continue
        
        data_gitw[obj] = {}
        subdirs = sorted(os.listdir(obj_path))
        for subd in subdirs:
            frames_dir = os.path.join(obj_path, subd, 'Frames')
            if not os.path.isdir(frames_dir):
                continue
            pred_dir = os.path.join(gitw_base, 'flow_pred_raft', obj, subd)
            if not os.path.isdir(pred_dir):
                continue
            
            frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png') or f.endswith('.jpg')])
            mse_list = []
            for i in range(len(frames)-1):
                img1_path = os.path.join(frames_dir, frames[i])
                img2_path = os.path.join(frames_dir, frames[i+1])
                flo_name = frames[i].replace('.png', '.flo').replace('.jpg', '.flo')
                flo_path = os.path.join(pred_dir, flo_name)
                if not os.path.isfile(flo_path):
                    continue

                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)
                flow_pred = read_flo_file(flo_path)

                warped = flow_warp(img1, flow_pred)
                mse = compute_mse(warped, img2)
                mse_list.append(mse)

            data_gitw[obj][subd] = mse_list

    # 5.1) Graphique MSE par index (pour chaque (obj, subd))
    for obj in data_gitw:
        for subd in data_gitw[obj]:
            mse_list = data_gitw[obj][subd]
            if not mse_list:
                continue

            indexes = range(len(mse_list))
            plt.figure(figsize=(8,5))
            plt.plot(indexes, mse_list, '-o', label='MSE')
            plt.title(f"GITW {obj}/{subd} - MSE vs. frame index")
            plt.xlabel("Frame index")
            plt.ylabel("MSE")
            plt.legend()
            plt.grid(True)
            plt.show()
            # plt.savefig(f"gitw_{obj}_{subd}_mse.png")

    # 5.2) Graphique MSE moyen par séquence
    #     On peut faire un bar plot par (obj,subd).
    #     x-axis => "obj-subd", y => MSE moyen.
    labels = []
    means = []
    for obj in data_gitw:
        for subd in data_gitw[obj]:
            mse_list = data_gitw[obj][subd]
            if mse_list:
                mean_mse = np.mean(mse_list)
                labels.append(f"{obj}-{subd}")
                means.append(mean_mse)

    if len(labels) > 0:
        x_pos = np.arange(len(labels))
        plt.figure(figsize=(8,5))
        plt.bar(x_pos, means, width=0.6, color='tomato')
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.ylabel("MSE")
        plt.title("Mean MSE by sequence (GITW)")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        # plt.savefig("gitw_mse_bar.png")


###############################################################################
# 6) Main
###############################################################################

def main():
    # Adaptez les chemins si besoin
    sintel_base = "/Users/corentinperdrizet/Documents/enseirb/3A/video/MPI-Sintel_selection"
    gitw_base   = "/Users/corentinperdrizet/Documents/enseirb/3A/video/GITW_selection"

    # 1. MPI-Sintel : EPE + AAE
    evaluate_sintel(sintel_base)

    # 2. GITW : MSE
    evaluate_gitw(gitw_base)

if __name__ == "__main__":
    main()
