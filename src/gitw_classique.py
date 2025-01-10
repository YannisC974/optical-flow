import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def all_image_paths_in_directory(root_dir):
    """
    Retourne la liste de tous les chemins d'images (png/jpg) 
    trouvés récursivement à partir de root_dir.
    """
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG")
    all_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(valid_exts):
                full_path = os.path.join(dirpath, f)
                all_paths.append(full_path)
    return sorted(all_paths)

def warp_image_with_flow(image, flow):
    """
    Déforme (warp) l'image 'image' selon le flot 'flow'.
    - image : np.ndarray de shape (H, W, 3) en BGR ou RGB
    - flow  : np.ndarray de shape (H, W, 2)
    
    Retourne l'image warping (même shape que 'image').
    """
    h, w = flow.shape[:2]
    
    # Création des map_x et map_y pour la fonction cv2.remap
    # Remarque : 
    #   flow[y,x,0] = déplacement horizontal (u)
    #   flow[y,x,1] = déplacement vertical   (v)
    #
    # Pour “warper” la frame2 vers la frame1, 
    # on veut dire que le pixel (x,y) de la frame1 vient de 
    # la position (x + u, y + v) dans la frame2.
    
    # base grid : coordonnées (x,y) pour chaque pixel
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # On ajoute le flot
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    
    # Remap : interpolation bilinéaire
    warped = cv2.remap(
        image, 
        map_x, 
        map_y, 
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0
    )
    return warped

def compute_mse(img1, img2):
    """
    Calcule la Mean Square Error (MSE) entre deux images de même dimension.
    - img1, img2 : np.ndarray (H, W, 3) en uint8 ou float32
    """
    # Convertir en float32 si besoin
    if img1.dtype != np.float32:
        img1 = img1.astype(np.float32)
    if img2.dtype != np.float32:
        img2 = img2.astype(np.float32)
        
    diff = img1 - img2
    mse_value = np.mean(diff ** 2)
    return mse_value

def main():
    # Chemin vers votre dossier GITW
    # Suppose qu'il contient plusieurs sous-dossiers (chaque sous-dossier = 1 séquence)
    gitw_root = "GITW_selection"
    
    # Pour stockage des résultats (pour un bar chart global)
    seq_names = []
    seq_mse_means = []

    # Lister tous les sous-dossiers (séquences)
    sequences = sorted([d for d in os.listdir(gitw_root) 
                        if os.path.isdir(os.path.join(gitw_root, d))])

    print(f"Trouvé {len(sequences)} séquences dans '{gitw_root}'.\n")

    for seq in sequences:
        seq_dir = os.path.join(gitw_root, seq)
        # Récupérer toutes les images (png/jpg) de la séquence
        frames = all_image_paths_in_directory(seq_dir)
        
        if len(frames) < 2:
            print(f"[AVERTISSEMENT] Séquence '{seq}' : pas assez d'images pour calculer le flot.")
            continue

        mse_values = []

        # Parcours de toutes les paires (frame_i, frame_i+1)
        for i in range(len(frames) - 1):
            img1_path = frames[i]
            img2_path = frames[i + 1]
            
            # Lecture
            frame1 = cv2.imread(img1_path)
            frame2 = cv2.imread(img2_path)
            
            if frame1 is None or frame2 is None:
                print(f"[ERREUR] Impossible de lire {img1_path} ou {img2_path}")
                continue
            
            # Conversion en niveaux de gris pour le flot optique
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calcul du flot optique (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Warping de frame2 vers frame1
            warped2 = warp_image_with_flow(frame2, flow)
            
            # Calcul de la MSE entre frame1 (originale) et frame2 compensée
            mse_val = compute_mse(frame1, warped2)
            mse_values.append(mse_val)

        # Affichage des résultats sur la séquence courante
        if mse_values:
            avg_mse = np.mean(mse_values)
            seq_names.append(seq)
            seq_mse_means.append(avg_mse)
            
            print(f"Séquence '{seq}': {len(mse_values)} paires traitées.")
            print(f"  MSE moyen = {avg_mse:.4f}\n")
            
            # --- Plot MSE pour la séquence ---
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(mse_values)+1), mse_values, marker='o')
            plt.title(f"MSE par paire - Séquence '{seq}'")
            plt.xlabel("Index de la paire")
            plt.ylabel("MSE")
            plt.tight_layout()
            plt.show()
        else:
            print(f"[AVERTISSEMENT] Séquence '{seq}': pas de paires valides après lecture.\n")

    # --- Bar chart global : MSE moyen par séquence ---
    if seq_names:
        plt.figure(figsize=(8, 4))
        plt.bar(seq_names, seq_mse_means, color='green', alpha=0.7)
        plt.title("MSE moyen par séquence - Dataset GITW")
        plt.xlabel("Séquence")
        plt.ylabel("MSE moyen")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune séquence valide n'a été traitée.")

if __name__ == "__main__":
    main()
