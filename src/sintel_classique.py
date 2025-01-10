import os
import glob
import struct
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_flo_file(file_path):
    """
    Lit un fichier .flo (format Middlebury/Sintel).
    Retourne un numpy array de shape (H, W, 2).
    """
    with open(file_path, 'rb') as f:
        magic = struct.unpack('f', f.read(4))[0]
        assert magic == 202021.25, "Invalid .flo file format!"
        
        width, height = struct.unpack('ii', f.read(8))
        data = np.fromfile(f, np.float32, count=2 * width * height)
        data = data.reshape((height, width, 2))
    return data

def compute_epe(flow_gt, flow_pred):
    """
    Calcule l'End-Point Error moyen (EPE) entre flow_gt et flow_pred.
    flow_gt et flow_pred : shape (H, W, 2).
    """
    error = np.linalg.norm(flow_gt - flow_pred, axis=2)
    return np.mean(error)

def compute_aae(flow_gt, flow_pred):
    """
    Calcule l'Average Angular Error (en radians) entre flow_gt et flow_pred.
    flow_gt et flow_pred : shape (H, W, 2).
    """
    dot_product = np.sum(flow_gt * flow_pred, axis=2)
    magnitude_gt = np.linalg.norm(flow_gt, axis=2)
    magnitude_pred = np.linalg.norm(flow_pred, axis=2)
    
    denom = (magnitude_gt * magnitude_pred) + 1e-12
    cos_theta = dot_product / denom
    
    # Éviter les erreurs numériques
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # en radians
    
    return np.mean(theta)

def main():
    # Modifier ces chemins si besoin
    image_root = "MPI-Sintel_selection/training/final"
    flow_root  = "MPI-Sintel_selection/training/flow"
    
    # Pour stocker les moyennes de chaque séquence (pour un bar chart global)
    seq_names = []
    seq_epe_means = []
    seq_aae_means = []

    # Lister les sous-dossiers du dossier "final" (chaque sous-dossier = une séquence)
    sequences = sorted([d for d in os.listdir(image_root) 
                        if os.path.isdir(os.path.join(image_root, d))])

    print(f"Trouvé {len(sequences)} séquences dans '{image_root}'.\n")

    for seq in sequences:
        seq_final_dir = os.path.join(image_root, seq)
        seq_flow_dir = os.path.join(flow_root, seq)

        frames = sorted(glob.glob(os.path.join(seq_final_dir, "*.png")))
        flows  = sorted(glob.glob(os.path.join(seq_flow_dir, "*.flo")))

        if not frames or not flows:
            print(f"[AVERTISSEMENT] Séquence '{seq}' : pas d'images ou pas de fichiers .flo.")
            continue

        num_pairs = min(len(frames) - 1, len(flows))
        if num_pairs <= 0:
            print(f"[AVERTISSEMENT] Séquence '{seq}' : pas de paires valides.")
            continue

        # Pour tracer EPE/AAE en fonction de la paire i
        epe_values = []
        aae_values = []

        for i in range(num_pairs):
            img1_path = frames[i]
            img2_path = frames[i + 1]
            flow_gt_path = flows[i]

            # Lecture des images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            if img1 is None or img2 is None:
                print(f"[ERREUR] Impossible de lire {img1_path} ou {img2_path}.")
                continue

            # Lecture du flow GT
            flow_gt = read_flo_file(flow_gt_path)
            
            # Conversion en niveaux de gris pour Farneback
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Calcul du flot optique prédit (Farneback)
            flow_pred = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Calcul des métriques
            epe = compute_epe(flow_gt, flow_pred)
            aae = compute_aae(flow_gt, flow_pred)

            epe_values.append(epe)
            aae_values.append(aae)

        # Tracé des métriques de la séquence (EPE / AAE)
        if epe_values and aae_values:
            # On calcule la moyenne
            avg_epe = np.mean(epe_values)
            avg_aae = np.mean(aae_values)

            print(f"Séquence '{seq}': {len(epe_values)} paires traitées.")
            print(f"  Average EPE = {avg_epe:.4f}")
            print(f"  Average AAE (radians) = {avg_aae:.4f}\n")

            # Sauvegarder pour le bar chart global
            seq_names.append(seq)
            seq_epe_means.append(avg_epe)
            seq_aae_means.append(avg_aae)

            # --- Plot “par séquence” ---
            # On crée une figure avec 2 sous-plots (EPE et AAE)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f"Métriques pour la séquence '{seq}'")

            # 1) EPE
            ax[0].plot(range(1, len(epe_values)+1), epe_values, marker='o', color='blue')
            ax[0].set_title("End-Point Error (EPE)")
            ax[0].set_xlabel("Index de la paire")
            ax[0].set_ylabel("EPE")

            # 2) AAE
            ax[1].plot(range(1, len(aae_values)+1), aae_values, marker='x', color='red')
            ax[1].set_title("Average Angular Error (radians)")
            ax[1].set_xlabel("Index de la paire")
            ax[1].set_ylabel("AAE (radians)")

            plt.tight_layout()
            plt.show()
        else:
            print(f"[AVERTISSEMENT] Séquence '{seq}': pas de paires valides après lecture.")

    # --- Bar chart global : EPE et AAE moyens de toutes les séquences ---
    if seq_names:
        # 1) EPE Global
        plt.figure(figsize=(8, 4))
        plt.bar(seq_names, seq_epe_means, color='blue', alpha=0.6)
        plt.title("Moyenne EPE par séquence")
        plt.xlabel("Séquence")
        plt.ylabel("EPE")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # 2) AAE Global
        plt.figure(figsize=(8, 4))
        plt.bar(seq_names, seq_aae_means, color='red', alpha=0.6)
        plt.title("Moyenne AAE (radians) par séquence")
        plt.xlabel("Séquence")
        plt.ylabel("AAE (radians)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
