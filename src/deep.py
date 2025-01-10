#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'RAFT'))


from core.raft import RAFT
from core.utils.utils import InputPadder

###############################################################################
# 1. Fonctions utilitaires (écriture/lecture du flot, visualisation, etc.)
###############################################################################

TAG_FLOAT = 202021.25  # "MAGIC NUMBER" pour le format .flo

def write_flo_file(filename, flow):
    """
    Enregistre un champ de flot optique 'flow' dans un fichier binaire .flo 
    conforme aux spécifications de MPI-Sintel.
    flow : numpy (H, W, 2)
    """
    assert flow.ndim == 3 and flow.shape[2] == 2, "Le flot doit avoir 2 canaux (u, v)."
    height, width = flow.shape[:2]
    with open(filename, 'wb') as f:
        f.write(np.array(TAG_FLOAT, dtype=np.float32).tobytes())
        f.write(np.array(width, dtype=np.int32).tobytes())
        f.write(np.array(height, dtype=np.int32).tobytes())
        f.write(flow[:,:,0].astype(np.float32).tobytes())
        f.write(flow[:,:,1].astype(np.float32).tobytes())


def flow_to_color(flow, clip_flow=None):
    """
    Convertit un flot (H,W,2) en une image couleur (BGR) pour visualisation.
    - 'clip_flow' : limite l'amplitude max du flot pour un meilleur rendu.
    """
    h, w = flow.shape[:2]
    u = flow[:,:,0]
    v = flow[:,:,1]

    if clip_flow is not None:
        u = np.clip(u, -clip_flow, clip_flow)
        v = np.clip(v, -clip_flow, clip_flow)

    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # Hue de 0 à 180
    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude*10, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


###############################################################################
# 2. Parcours des répertoires pour récupérer les paires d'images
###############################################################################

def get_frame_pairs_sintel(sintel_base):
    """
    Récupère la liste des paires (img1, img2) consécutives dans
    MPI-Sintel_selection/training/[clean|final]/<sequence>.
    Retour : [(img1, img2, out_flo, out_visu), ...]
    """
    pairs = []
    subfolders = ['clean', 'final']  # ou ['clean'], ou ['final'] au choix

    for subf in subfolders:
        seq_dir = os.path.join(sintel_base, 'training', subf)
        if not os.path.isdir(seq_dir):
            continue
        
        sequences = sorted(os.listdir(seq_dir))
        for seq_name in sequences:
            seq_path = os.path.join(seq_dir, seq_name)
            if not os.path.isdir(seq_path):
                continue
            
            frames = sorted([f for f in os.listdir(seq_path) 
                             if f.endswith('.png') or f.endswith('.jpg')])

            for i in range(len(frames) - 1):
                img1 = os.path.join(seq_path, frames[i])
                img2 = os.path.join(seq_path, frames[i+1])

                # Sortie : flow_pred_raft/clean/sequence/frame_0001.flo
                out_flo = os.path.join(
                    sintel_base, 'training', 'flow_pred_raft',
                    subf, seq_name,
                    frames[i].replace('.png', '.flo').replace('.jpg', '.flo')
                )
                out_visu = out_flo.replace('.flo', '_visu.png')
                pairs.append((img1, img2, out_flo, out_visu))

    return pairs


def get_frame_pairs_gitw(gitw_base):
    """
    Récupère la liste des paires (img1, img2) dans GITW_selection/<obj>/<variation>/Frames.
    Retour : [(img1, img2, out_flo, out_visu), ...]
    """
    pairs = []
    objects = sorted(os.listdir(gitw_base))

    for obj in objects:
        obj_path = os.path.join(gitw_base, obj)
        if not os.path.isdir(obj_path):
            continue
        
        subdirs = sorted(os.listdir(obj_path))
        for subd in subdirs:
            frames_dir = os.path.join(obj_path, subd, 'Frames')
            if not os.path.isdir(frames_dir):
                continue
            
            frames = sorted([f for f in os.listdir(frames_dir)
                             if f.endswith('.png') or f.endswith('.jpg')])
            
            for i in range(len(frames) - 1):
                img1 = os.path.join(frames_dir, frames[i])
                img2 = os.path.join(frames_dir, frames[i+1])

                # Sortie : flow_pred_raft/obj/subd/frame_0001.flo
                out_flo = os.path.join(
                    gitw_base, "flow_pred_raft",
                    obj, subd,
                    frames[i].replace('.png', '.flo').replace('.jpg', '.flo')
                )
                out_visu = out_flo.replace('.flo', '_visu.png')
                pairs.append((img1, img2, out_flo, out_visu))

    return pairs


###############################################################################
# 3. Fonctions PyTorch : chargement RAFT et calcul du flot
###############################################################################


def load_raft_model(weights_path, device='cuda'):
    from argparse import Namespace
    args = Namespace(
        small=False,       # ou True si vous avez un checkpoint small
        mixed_precision=False,
        alternate_corr=False,
    )

    model = RAFT(args)
    model.to(device)

    # Charger le checkpoint
    ckpt = torch.load(weights_path, map_location=device)

    state_dict = {}
    for k, v in ckpt.items():
        # Si la clé commence par 'module.', on la retire
        new_k = k.replace('module.', '')
        state_dict[new_k] = v

    # Charger le state_dict
    model.load_state_dict(state_dict, strict=False)  # strict=False si vous voulez ignorer d'éventuelles clés manquantes
    model.eval()
    return model



@torch.no_grad()
def inference_raft(model, image1, image2, device='cuda'):
    """
    Calcule le flot optique (u,v) entre 2 images (H,W,3) BGR ou RGB, en numpy.
    - model : instance RAFT déjà chargée
    - image1, image2 : np.array (H, W, 3), 0-255
    - Retourne : flow (H, W, 2), np.float32
    """
    # Convertir en tenseurs PyTorch, normaliser
    # RAFT attend un tenseur [1,3,H,W] en dtype float32, canal RGB dans [0,1].
    # On va donc passer BGR->RGB si besoin.
    # OpenCV lit en BGR, RAFT attend en RGB => on inverse si on veut être cohérent.
    img1_t = torch.from_numpy(image1[:,:,::-1].copy()).permute(2,0,1).float()
    img2_t = torch.from_numpy(image2[:,:,::-1].copy()).permute(2,0,1).float()

    img1_t = img1_t[None].to(device) / 255.0  # shape = (1,3,H,W)
    img2_t = img2_t[None].to(device) / 255.0

    # RAFT a besoin qu'on "padd" les images pour que la taille soit multiple de 8
    padder = InputPadder(img1_t.shape)
    img1_t, img2_t = padder.pad(img1_t, img2_t)

    # Inférence (20 itérations typiquement)
    flow_low, flow_up = model(img1_t, img2_t, iters=20, test_mode=True)
    # flow_up => [1, 2, H, W]

    # On retire le padding
    flow_up = padder.unpad(flow_up)[0]  # shape=(2, H, W)

    # Conversion en numpy
    flow_np = flow_up.permute(1,2,0).cpu().numpy()  # (H, W, 2)
    return flow_np


###############################################################################
# 4. Programme principal
###############################################################################

def main():
    # 4.1. Paramètres : chemins
    sintel_path = "MPI-Sintel_selection"
    gitw_path   = "GITW_selection"
    weights     = "raft-sintel.pth"  # nom du checkpoint RAFT (à adapter)

    # Vérifie si un GPU est dispo, sinon CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # 4.2. Charger le modèle RAFT
    print("[INFO] Chargement du modèle RAFT...")
    model = load_raft_model(weights, device=device)

    # 4.3. Récupérer les paires d’images
    pairs_sintel = get_frame_pairs_sintel(sintel_path)
    pairs_gitw   = get_frame_pairs_gitw(gitw_path)
    print(f"[INFO] Nombre de paires (Sintel) = {len(pairs_sintel)}")
    print(f"[INFO] Nombre de paires (GITW)   = {len(pairs_gitw)}")

    # 4.4. Itérer sur chaque paire
    all_pairs = [('Sintel', p) for p in pairs_sintel] + \
                [('GITW',   p) for p in pairs_gitw]

    for dataset_name, (img1, img2, out_flo, out_visu) in all_pairs:
        # Lecture images
        frame1 = cv2.imread(img1)
        frame2 = cv2.imread(img2)
        if frame1 is None or frame2 is None:
            print(f"[ERREUR] Impossible de lire {img1} ou {img2}")
            continue

        # Création du répertoire de sortie si besoin
        os.makedirs(os.path.dirname(out_flo), exist_ok=True)

        # Calcul du flow
        try:
            flow = inference_raft(model, frame1, frame2, device=device)
        except Exception as e:
            print(f"[ERREUR] RAFT n'a pas pu calculer le flot : {e}")
            continue

        # Écriture du .flo
        write_flo_file(out_flo, flow)

        # Visualisation couleur
        color_flow = flow_to_color(flow, clip_flow=25.0)
        out_visu_dir = os.path.dirname(out_visu)
        os.makedirs(out_visu_dir, exist_ok=True)
        cv2.imwrite(out_visu, color_flow)

        print(f"[OK] {dataset_name} : {os.path.basename(img1)} => {out_flo}")

    print("[FINI] Tout est traité !")


if __name__ == "__main__":
    main()
