import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import label, find_objects, center_of_mass, uniform_filter1d
from scipy.signal import find_peaks, peak_prominences
from matplotlib.patches import Ellipse as MplEllipse
from photutils.isophote import EllipseGeometry, Ellipse
from skimage import measure
import warnings
from scipy.fft import fft, fftfreq

#image loading

warnings.filterwarnings("ignore", category=RuntimeWarning)

image_folder = 'images'
fits_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.fits')])

if not fits_files:
    raise FileNotFoundError("No FITS files found in the 'images' folder.")

#Starting for loop to go through multiple images

for idx, file in enumerate(fits_files, 1):
    try:

        #Load and normalize all images

        path = os.path.join(image_folder, file)
        print(f"\n[PROCESSING {idx}/{len(fits_files)}] {file}")

        data = fits.getdata(path)
        while data.ndim > 2:
            data = data[0]

        #Handling NA values
        #To be noted : Can't use .Na function like usual, the numpy .na functions is good for tables
            
        data = np.nan_to_num(data)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Marks pixels brighter than 0.2 as True. Basically creating a binary mask idetifying disconnected bright regions. 
        # Disks are often centrally peaked, so this threshold focuses on emission above background levels.

        threshold = 0.2
        mask = data > threshold
        labeled, num_features = label(mask)
        regions = find_objects(labeled)
        crop_applied = False #in case we later have to crop the image

        #IMPORTANT - This snippet does a few things, firs it finds all the centroids using the threshold pixels. The pixel with max intensity is idetified as the center.
        #Finds the label number of the region, marks it as primary.
        #Finds a center, basically the main object. And thus crops around this spot, ignoring the 

        if num_features >= 2:
            centers = center_of_mass(mask, labeled, index=range(1, num_features + 1)) #Code to calculate the centre of mass
            y0, x0 = np.unravel_index(np.argmax(data), data.shape)
            main_label = labeled[y0, x0]
            main_center = centers[main_label - 1]

            for i, center in enumerate(centers):
                if i + 1 == main_label:
                    continue
                dist = np.linalg.norm(np.array(center) - np.array(main_center))

                #Code to crop the images, this is because in some images there are certain bright blobs outside the region of the star, that might be considered as the protostars region. 
                #So to ensure that is avoided, the images are getting cropped here if necessary. 
                #


                if dist > 300:
                    region_slice = regions[main_label - 1]
                    margin = 50
                    y_start = max(region_slice[0].start - margin, 0)
                    y_end = min(region_slice[0].stop + margin, data.shape[0])
                    x_start = max(region_slice[1].start - margin, 0)
                    x_end = min(region_slice[1].stop + margin, data.shape[1])
                    data = data[y_start:y_end, x_start:x_end]
                    crop_applied = True
                    print(f"[INFO] Cropped to main disk (distance = {dist:.1f})")
                    break

        y0, x0 = np.unravel_index(np.argmax(data), data.shape)
        ellipse_failed = False
        try:
            geometry = EllipseGeometry(x0=x0, y0=y0, sma=5, eps=0.3, pa=0.0)
            ellipse = Ellipse(data, geometry)
            isolist = ellipse.fit_image()
            if len(isolist) == 0:
                raise ValueError("no valid isophotes found")
            last_iso = isolist[-1]
            x0_ell, y0_ell = last_iso.x0, last_iso.y0
            eps, theta = last_iso.eps, last_iso.pa
            q = 1 - eps
        except Exception as e:
            print(f"[WARNING] Ellipse fitting failed for {file}: {e}")
            ellipse_failed = True
            x0_ell, y0_ell = x0, y0
            eps, theta, q = 0, 0, 1

        y, x = np.indices(data.shape)
        x_rel = x - x0_ell
        y_rel = y - y0_ell
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x_rel * cos_t + y_rel * sin_t
        y_rot = -x_rel * sin_t + y_rel * cos_t
        r_elliptical = np.sqrt(x_rot**2 + (y_rot / q)**2).astype(int)

        tbin = np.bincount(r_elliptical.ravel(), data.ravel())  # Total brightness per radius
        nr = np.bincount(r_elliptical.ravel())                  # Number of pixels per radius
        radial_profile = tbin / nr                              # Average brightness per radius
        valid = nr > 0
        radial_profile[valid] = tbin[valid] / nr[valid]
        smoothed = uniform_filter1d(radial_profile, size=5)

        min_val = np.min(smoothed)
        tolerance = 0.005
        window = 20
        for cutoff in range(50, len(smoothed) - window):
            if np.all(np.abs(smoothed[cutoff:cutoff + window] - min_val) < tolerance):
                break
        else:
            cutoff = len(smoothed)

        peaks, _ = find_peaks(smoothed[:cutoff], prominence=0.005, distance=10)

        from scipy.stats import kurtosis


        clumpiness_index = kurtosis(smoothed[:cutoff])

        if len(peaks) > 2:
            spacing_diffs = np.diff(peaks)
            regularity = np.std(spacing_diffs) / np.mean(spacing_diffs)
        else:
            regularity = 0
        os.makedirs("profiles", exist_ok=True)
        profile_path = f"profiles/{file.replace('.fits', '_profile.csv')}"
        np.savetxt(profile_path, smoothed[:cutoff], delimiter=",")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        zoom_size = 150
        y0_vis, x0_vis = np.unravel_index(np.argmax(data), data.shape)
        signal_above = np.where(smoothed > (min_val + 0.01))[0]
        estimated_radius = signal_above[-1] if len(signal_above) > 0 else 0
        should_zoom = estimated_radius < 0.2 * min(data.shape)

        if should_zoom:
            ymin = max(0, y0_vis - zoom_size)
            ymax = min(data.shape[0], y0_vis + zoom_size)
            xmin = max(0, x0_vis - zoom_size)
            xmax = min(data.shape[1], x0_vis + zoom_size)
            display_data = data[ymin:ymax, xmin:xmax]
            cx_shift, cy_shift = xmin, ymin
            zoom_note = " (zoomed-in)"
        else:
            display_data = data
            cx_shift, cy_shift = 0, 0
            zoom_note = ""

        im = ax1.imshow(display_data, origin='lower', cmap='inferno')
        ax1.set_title(f"{file}{zoom_note}")

        mean_prominence = 0
        spacing_std = 0
        num_peaks = len(peaks)
        if num_peaks > 0:
            prominences = peak_prominences(smoothed[:cutoff], peaks)[0]
            mean_prominence = np.mean(prominences)
            if num_peaks > 1:
                spacing_std = np.std(np.diff(peaks))

        if not ellipse_failed and len(peaks) >= 2 and mean_prominence > 0.01:
            for radius in peaks:
                cx = x0_ell - cx_shift
                cy = y0_ell - cy_shift
                e = MplEllipse((cx, cy), 2 * radius, 2 * radius * q,
                               angle=np.degrees(theta), edgecolor='cyan',
                               facecolor='none', linestyle='--', lw=1)
                ax1.add_patch(e)
        else:
            contour_levels = np.linspace(0.1, 0.9, 6)
            for level in contour_levels:
                contours = measure.find_contours(display_data, level=level)
                for contour in contours:
                    if len(contour) > 100:
                        ax1.plot(contour[:, 1], contour[:, 0], linestyle='--', color='cyan', linewidth=1)

        fig.colorbar(im, ax=ax1, fraction=0.046)
        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")

        ax2.plot(smoothed[:cutoff], color='deepskyblue', lw=2, label='Smoothed Profile')
        ax2.plot(peaks, smoothed[peaks], "ro", label='Detected Rings')
        ax2.vlines(peaks, ymin=0, ymax=smoothed[peaks], colors='r', linestyles='dashed', alpha=0.5)
        ax2.set_xlim(0, cutoff)
        ax2.set_title("Elliptical Radial Brightness Profile")
        ax2.set_xlabel("Elliptical Radius (pixels)")
        ax2.set_ylabel("Average Intensity")
        ax2.grid(alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        plt.show()
        

        h, w = data.shape
        left_flux = np.sum(data[:, :w // 2])
        right_flux = np.sum(data[:, w // 2:])
        asymmetry_index = abs(left_flux - right_flux) / (left_flux + right_flux + 1e-8)

        suggested_label = "weird"
        if num_peaks == 0:
            if asymmetry_index < 0.05 and mean_prominence < 0.01:
                suggested_label = "smooth-compact"
            elif asymmetry_index < 0.1:
                suggested_label = "smooth-extended"
            elif np.max(data) < 0.1:
                suggested_label = "smooth-faint"
            else:
                suggested_label = "smooth-offset"
        elif num_peaks >= 2:
            if mean_prominence > 0.03 and spacing_std > 10:
                suggested_label = "ring-multiple"
            elif mean_prominence < 0.02:
                suggested_label = "ring-weak"
            elif np.mean(peaks) < 40:
                suggested_label = "ring-inner-only"
            elif spacing_std > 20:
                suggested_label = "ring-wide-spacing"
        elif asymmetry_index > 0.15:
            if asymmetry_index > 0.4:
                suggested_label = "asym-bright-lobe"
            elif 0.2 < asymmetry_index <= 0.4:
                suggested_label = "asym-eccentric"
            elif h > 100 and (np.sum(data[h//2:, :] < 0.1) > 0.3 * w):
                suggested_label = "asym-tail"
            elif np.std(data[:, w//2]) > 0.1:
                suggested_label = "asym-warped"

        # Improved weird classification logic
        if suggested_label == "weird":
            if spacing_std > 30 and num_peaks > 1 and mean_prominence > 0.01:
                suggested_label = "weird-broken-ring"
            elif np.std(data[:h//3]) < 0.05 and np.std(data[h//3:]) > 0.2:
                suggested_label = "weird-shadowed"
            elif num_features >= 2 and crop_applied:
                suggested_label = "weird-multiple-stars"
            elif "spiral" in file.lower():
                suggested_label = "weird-spiral"
            elif mean_prominence < 0.005 and asymmetry_index < 0.05:
                suggested_label = "smooth-faint"

        suggested_label = "weird"  # default fallback

# Smooth categories
        if num_peaks == 0:
            if asymmetry_index < 0.05 and mean_prominence < 0.01:
                suggested_label = "smooth-compact"
            elif np.max(data) < 0.1:
                suggested_label = "smooth-faint"
            elif asymmetry_index < 0.1:
                suggested_label = "smooth-extended"
            else:
                suggested_label = "smooth-offset"

# Ring categories
        elif num_peaks >= 2:
            if mean_prominence > 0.03 and spacing_std > 10:
                suggested_label = "ring-multiple"
            elif mean_prominence > 0.01 and spacing_std < 15:
                suggested_label = "ring-weak"
            elif np.mean(peaks) < 40:
                suggested_label = "ring-inner-only"
            elif spacing_std > 20:
                suggested_label = "ring-wide-spacing"

# Asymmetric
        elif asymmetry_index > 0.15:
            if asymmetry_index > 0.4:
                suggested_label = "asym-bright-lobe"
            elif 0.2 < asymmetry_index <= 0.4:
                suggested_label = "asym-eccentric"
            elif h > 100 and (np.sum(data[h//2:, :] < 0.1) > 0.3 * w):
                suggested_label = "asym-tail"
            elif np.std(data[:, w//2]) > 0.1:
                suggested_label = "asym-warped"

# Fallback weird categories
        if suggested_label.startswith("weird") or suggested_label == "weird":
            if spacing_std > 30 and num_peaks > 1 and mean_prominence > 0.01:
                suggested_label = "weird-broken-ring"
            elif np.std(data[:h//3]) < 0.05 and np.std(data[h//3:]) > 0.2:
                suggested_label = "weird-shadowed"
            elif num_features >= 2 and crop_applied:
                suggested_label = "weird-multiple-stars"
            elif "spiral" in file.lower():
                suggested_label = "weird-spiral"
            elif mean_prominence < 0.005 and asymmetry_index < 0.05:
                suggested_label = "smooth-faint"
            else:
                suggested_label = "smooth-extended"  # final fallback

        print(f"[SUGGESTED LABEL] → {suggested_label}")
        disk_label = input("Label this disk (press Enter to accept suggestion or type your own): ").strip().lower()
        if disk_label == "":
            disk_label = suggested_label

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping the script manually. Exiting.")
        break
    except Exception as err:
      print(f"[ERROR] Skipping {file} due to error: {err}")
    continue

df = pd.read_csv("disk_summary.csv")
profile_folder = "profiles"  
output_folder = "overlaid_profiles"

from scipy.fft import fft, fftfreq

for label, group in df.groupby("label"):
    plt.figure(figsize=(10, 6))
    
    for _, row in group.iterrows():

        base = os.path.splitext(row["filename"])[0]
        profile_path = os.path.join(profile_folder, f"{base}_profile.csv")
        print(f"🔍 Checking: {profile_path}")

        if os.path.exists(profile_path):
            print(f"✅ Found: {profile_path}")
            profile = pd.read_csv(profile_path, header=None).squeeze("columns")
        else:
            print(f"❌ Missing: {profile_path}")

        base = os.path.splitext(row["filename"])[0]
        profile_path = os.path.join(profile_folder, f"{base}_profile.csv")
        
        if os.path.exists(profile_path):
            profile = pd.read_csv(profile_path, header=None).squeeze("columns")
            profile_np = profile.to_numpy() 
            N = len(profile_np)
            yf = np.abs(fft(profile_np))
            xf = fftfreq(N, d=1)[:N // 2]
            plt.plot(xf, yf[:N // 2], label=base)

    plt.title(f"Fourier Spectrum (Radial) - {label}")
    plt.xlabel("Frequency (1/pixel)")
    plt.ylabel("Amplitude")
    plt.legend(title="Disk")
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, 0.1)       
    plt.ylim(0, 50)    
    plt.savefig(os.path.join(output_folder, f"{label}_fourier.png"))
    plt.close()
