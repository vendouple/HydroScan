import os
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PIL import Image
from io import BytesIO

# ✅ CONFIG (edited save dir)
CATEGORIES = {
    "Water": ["water", "ocean", "lake", "river", "underwater"],
    "NotWater": ["scenery", "mountain", "forest", "desert", "urban"],
}
MAX_IMAGES = 2
SAVE_DIR = "InHouseModelTraining\\Classification\\Images\\Unsorted"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

SOURCES = {
    "Pexels": "https://www.pexels.com/search/{}/",
    "Pixabay": "https://pixabay.com/images/search/{}/",
    "Unsplash": "https://unsplash.com/s/photos/{}",
    "Freepik": "https://www.freepik.com/free-photos-vectors/{}",
    "StockSnap": "https://stocksnap.io/search/{}",
    "Burst": "https://burst.shopify.com/photos/search?utf8=✓&q={}",
    "Kaboompics": "https://kaboompics.com/gallery?search={}",
    "ISORepublic": "https://isorepublic.com/?s={}",
    "Picjumbo": "https://picjumbo.com/?s={}",
    "Reshot": "https://www.reshot.com/search/{}/",
}

# thresholds to skip placeholders
MIN_IMAGE_BYTES = 5 * 1024  # skip files smaller than 5 KB (likely placeholders)
MIN_IMAGE_SIDE = 100  # skip images with width or height < 100 px

session = requests.Session()
session.headers.update(HEADERS)


def parse_srcset(srcset):
    """
    Parse a srcset string and return the URL of the largest candidate.
    Handles formats like:
      url1 300w, url2 600w
      url1 1x, url2 2x
    """
    if not srcset:
        return None
    parts = [p.strip() for p in srcset.split(",") if p.strip()]
    best = None
    best_value = -1
    for p in parts:
        # each part: "<url> <descriptor>"
        tokens = p.split()
        url = tokens[0]
        descriptor = tokens[1] if len(tokens) > 1 else None
        value = 0
        if descriptor and descriptor.endswith("w"):
            try:
                value = int(descriptor[:-1])
            except:
                value = 0
        elif descriptor and descriptor.endswith("x"):
            try:
                value = float(descriptor[:-1]) * 1000
            except:
                value = 0
        else:
            value = 0
        if value > best_value:
            best_value = value
            best = url
    return best or (parts[-1].split()[0] if parts else None)


def extract_image_url(img_tag, base_url):
    """
    Try multiple attributes in order to get the 'real' image url.
    - data-srcset -> parse, pick largest
    - data-src
    - srcset -> parse
    - src
    - style background-image (rare)
    """
    candidates = []
    # prefer data-srcset / data-src which lazyloaders use
    for attr in (
        "data-srcset",
        "data-src",
        "data-lazy-src",
        "data-original",
        "data-srcset-1280",
    ):
        val = img_tag.get(attr)
        if val:
            if "srcset" in attr:
                parsed = parse_srcset(val)
                if parsed:
                    candidates.append(parsed)
            else:
                candidates.append(val)

    # then srcset and src
    if img_tag.get("srcset"):
        parsed = parse_srcset(img_tag.get("srcset"))
        if parsed:
            candidates.append(parsed)
    if img_tag.get("src"):
        candidates.append(img_tag.get("src"))

    # fallback: images sometimes embedded in style attribute as background-image
    style = img_tag.get("style", "")
    if "background-image" in style:
        # crude extract
        import re

        m = re.search(r"url\(([^)]+)\)", style)
        if m:
            url = m.group(1).strip("'\" ")
            candidates.append(url)

    # normalize and pick first valid absolute URL
    for src in candidates:
        if not src:
            continue
        if src.startswith("data:"):
            # skip inline data placeholders
            continue
        # make absolute
        src_abs = urljoin(base_url, src)
        # if protocol-relative //example.com/img.jpg
        if src_abs.startswith("//"):
            parsed = urlparse(base_url)
            src_abs = parsed.scheme + ":" + src_abs
        return src_abs
    return None


def is_likely_placeholder(resp_content, resp_headers, img):
    # check content-length or file bytes
    content_len = len(resp_content) if resp_content is not None else 0
    if content_len < MIN_IMAGE_BYTES:
        return True
    # check content-type header
    ctype = resp_headers.get("Content-Type", "")
    if not ctype.startswith("image/"):
        return True
    # try to open with PIL to check dimensions
    try:
        image = Image.open(BytesIO(resp_content))
        w, h = image.size
        if w < MIN_IMAGE_SIDE or h < MIN_IMAGE_SIDE:
            return True
    except Exception:
        # can't open -> treat as placeholder/corrupt
        return True
    return False


def save_as_jpg(img_data, save_path):
    try:
        img = Image.open(BytesIO(img_data)).convert("RGB")
        img.save(save_path, format="JPEG", quality=95)
        print(f"Saved: {save_path} ({img.size[0]}x{img.size[1]})")
    except Exception as e:
        print(f"Failed to convert image to JPEG: {e}")


def scrape_images(category, keywords):
    folder = os.path.join(SAVE_DIR, category)
    os.makedirs(folder, exist_ok=True)
    count = 0

    for keyword in keywords:
        for site, url_template in SOURCES.items():
            search_url = url_template.format(keyword)
            try:
                print(f"Scraping {site} for '{keyword}' -> {search_url}")
                # set Referer to the search page so CDNs think the request originates from that page
                page_resp = session.get(
                    search_url, timeout=12, headers={"Referer": search_url}
                )
                page_resp.raise_for_status()
                soup = BeautifulSoup(page_resp.text, "html.parser")
                img_tags = soup.find_all("img")

                for img in img_tags:
                    if count >= MAX_IMAGES:
                        break
                    src = extract_image_url(img, search_url)
                    if not src:
                        continue
                    # skip inline data URIs
                    if src.startswith("data:"):
                        continue

                    try:
                        # request with referer and random short delay to reduce blocking
                        headers = {"Referer": search_url}
                        resp = session.get(src, headers=headers, timeout=12)
                        # if redirect to an HTML page (anti-bot), skip
                        content_type = resp.headers.get("Content-Type", "")
                        if not content_type.startswith("image/"):
                            # sometimes src is a landing page; skip
                            continue
                        content = resp.content

                        # skip tiny placeholders
                        if is_likely_placeholder(content, resp.headers, img):
                            # optionally try next candidate by continuing, but for simplicity skip
                            continue

                        fname = f"{category}_{keyword}_{count}.jpg"
                        save_path = os.path.join(folder, fname)
                        save_as_jpg(content, save_path)
                        count += 1

                    except Exception as e_img:
                        # network or image open error for this src
                        # print minimal debug to avoid leaking lots of logs
                        # print(f"Failed to download {src}: {e_img}")
                        continue

                # politeness: small random sleep (reduces pattern-like scraping)
                time.sleep(1 + random.random() * 2)
                if count >= MAX_IMAGES:
                    break
            except Exception as e:
                print(f"Error scraping {site}: {e}")
            if count >= MAX_IMAGES:
                break
        if count >= MAX_IMAGES:
            break


# ✅ RUN SCRAPER
for cat, keywords in CATEGORIES.items():
    scrape_images(cat, keywords)
