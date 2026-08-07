import hashlib
import os

import requests


# Model repo: celebA classifier, GAN safetensors, per-attribute classifiers.
_MODEL_BASE_URL = (
    "https://huggingface.co/ibm-research/AIX360-CEM-MAF-USECASE/resolve/main/"
)

# Dataset repo: per-id demo images. Note the flat layout — no `celeba_data/`
# prefix on the remote URL — while the local destination still lives under
# aix360/data/celeba_data/. dwnld_celebA_data handles the asymmetry.
_DATA_BASE_URL = (
    "https://huggingface.co/datasets/ibm-research/AIX360-CEM-MAF-data/resolve/main/"
)

# SHA-256 of every file the downloader is allowed to fetch. Computed from the
# trusted local copies. Any file whose bytes don't match its pinned digest is
# rejected — defends against tampering, MITM on the transport, and the remote
# host being reassigned to an attacker.
_EXPECTED_SHA256 = {
    # Attribute classifiers (15 attributes × [ckpt, model.json, weights.h5])
    "simple_Attractive.ckpt":            "13274345eddee136bade8d899000c236d4cff42e444959f8021fc7e7c9fdbd82",
    "simple_Bags_Under_Eyes.ckpt":       "9725387889bfd4485226e1a9f0bf55c204ab25be3f04a545a9774e689832b911",
    "simple_Bangs.ckpt":                 "a0d17716041005ca3d3a44936a9e2e7354863a10abf3007d1286cd9fa4ba0f6e",
    "simple_Black_Hair.ckpt":            "a668ed53bc42a0d3cf105116c83e6525d3b219935d668dcd849c48c0643eae40",
    "simple_Blond_Hair.ckpt":            "74a3ec9abb1bd6cd933800b86ba6e96f0d161d291cb0b5b30930d4ea4843af6c",
    "simple_Brown_Hair.ckpt":            "a6354cb8786278ba76779d7d1b98711c9d207a33295626db9a9b7617603647c5",
    "simple_Gray_Hair.ckpt":             "90a6a5eaf035938c4490198594880cc74cf60dad2fcf36b45ae5d8eb52b88a9a",
    "simple_Heavy_Makeup.ckpt":          "df9ee2993289ab59a7edb9b36e7ad85bb520cbfae9744107d90be00fec0ca907",
    "simple_High_Cheekbones.ckpt":       "c2e92a764062972bba54ecb89d847f8fc73e86fd31ab208968a33361e487a880",
    "simple_Narrow_Eyes.ckpt":           "d5819f8b1e2f5012cae403536fdc6f1b9ea1b77d06640765ba67ecdc4efe8f5a",
    "simple_Oval_Face.ckpt":             "3b9bca1fce3f25e044cc25bb165f8e506898896c5bb96c6665c890be15848bca",
    "simple_Pointy_Nose.ckpt":           "6fc01968b78529b9331c6037adfa2c20f1041c006ad62d9a9a8ad5f2828341c0",
    "simple_Smiling.ckpt":               "1283caa5d64a45de0bef904a689cf33c787d4b1293e6f3741cc1ff8529d2cfaf",
    "simple_Wearing_Lipstick.ckpt":      "baf7f907e3a667b0716218f95e04f7dd984b660b9b738e88eaaa3a23765d6572",
    "simple_Young.ckpt":                 "10bb888a182945b84ba699de5a69caceeee381aed7bfa1c7bfc4a1d91a5d4298",
    "simple_Attractive_model.json":      "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Bags_Under_Eyes_model.json": "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Bangs_model.json":           "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Black_Hair_model.json":      "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Blond_Hair_model.json":      "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Brown_Hair_model.json":      "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Gray_Hair_model.json":       "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Heavy_Makeup_model.json":    "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_High_Cheekbones_model.json": "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Narrow_Eyes_model.json":     "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Oval_Face_model.json":       "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Pointy_Nose_model.json":     "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Smiling_model.json":         "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Wearing_Lipstick_model.json":"8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Young_model.json":           "8e798f7099fb470eebfa0f160820e113e9742042cd36b370443fcce568e64577",
    "simple_Attractive_weights.h5":      "cedab92ffc587f43c7b90ec4a79feecdf44df125ef44612325dde36822a514a5",
    "simple_Bags_Under_Eyes_weights.h5": "35cba0bd61550c965c157f8ebfbf1b167149aec9f852768d4b63e40ef741df2b",
    "simple_Bangs_weights.h5":           "c1b964ae600baab69a93ad629775c74c39dd0dd58c900746313a60e6313da500",
    "simple_Black_Hair_weights.h5":      "bf9b831ebfbf8f2f78398fd180ea75b908b3774accc9f5c740e572735e722033",
    "simple_Blond_Hair_weights.h5":      "ee613e6a074f091ac88c4e7f46116f139ffa3195f0c043a9e28545a71dcdbae7",
    "simple_Brown_Hair_weights.h5":      "9f371af15e2a2af0a5e359b7a1dfe4ed3087f7ee12235e68d4eb1863ab9f454d",
    "simple_Gray_Hair_weights.h5":       "07672b357e2de59c259f32f534d00e95b3b32a856e9bd49c48d068098e8121cb",
    "simple_Heavy_Makeup_weights.h5":    "b0755d2be649e3cd3a4709b23615ec4351d2d71e8d73d4b131742359577dd8fe",
    "simple_High_Cheekbones_weights.h5": "e2804cb12c2cd5ecdc4cd029f780108ff978b2ea4c6c6836d2b241d5c9307571",
    "simple_Narrow_Eyes_weights.h5":     "399aca10a791e02ffc80fb099b6ef4cec70478f5bb49fa19a6b1ee256dd27f23",
    "simple_Oval_Face_weights.h5":       "37780e173d52eed8727600523ef3662f7bcea556a4a987db44112d130388e160",
    "simple_Pointy_Nose_weights.h5":     "a4b2edebb86d2a95f6bb596fb0ebcd6ccc9dd8722e41252328404fdef951a22d",
    "simple_Smiling_weights.h5":         "5c4a03a9811c966345f35b5ac8ebe42a8b0f293bb3c0731439632caa7e7a921b",
    "simple_Wearing_Lipstick_weights.h5":"abe8c48105f07db7febcab76ceaa76a0e70c630e3e44d7c7a2a60923455f16db",
    "simple_Young_weights.h5":           "719961114092f23b9832bd28540f316b3b1f4dcde07441852c2077f47d916edd",
    # CelebA prediction model
    "celebA": "f2b2af6f4e57b7c80d06d8c68d38d4a9dacb2c651522aa935c7e2aeee0f01dee",
    # Image data — only ids the HF dataset repo actually hosts.
    "2_img.npy":     "5464b7f578b5c2a0508da2c56b95e7c798a1b17d102525f8fc2b0417e13d7400",
    "2_latent.npy":  "22cfab538b10f0c00c4cd1d847c0ccaadde3622b98bc5def11f33a8971e2502d",
    "2_img.png":     "fbe7f3079f7875b5f67b53c4fb070e82a89e4eaeecbfa5b48a56461e24e20a8d",
    "3_img.npy":     "d051c94e96cfd9335c2a39d15407b5467bc71e6c5eb2b678c6f48f8457297aed",
    "3_latent.npy":  "258c40934612f5463c03f5f07135cd2e42fefa74301a2f781249270c85fccd4a",
    "3_img.png":     "71b22743260357d5b9caf3280215f1c3aca1448fad5ce73a5a8b89bcfe18c7f6",
    "4_img.npy":     "4816e45d7c2820109d1eb1579800078d5be10f641fb0bcc7ac19eb59d26d0afe",
    "4_latent.npy":  "d8294e45d2021dda6b330e6f1c163a66478476986b7207c6caa0824c65b893cb",
    "4_img.png":     "1f751d87e8d9a28d6a6ba19b75088451a450263f99da4cafdf776c9efc2439d1",
    "9_img.npy":     "ffc51e17cd62a4290738a38dbd7b42c088c73ba4831e91b13a263e20ac9accd6",
    "9_latent.npy":  "87a0b285ba197f33b4d5824dde04995e82eb31ffe26d171d7638247865a91376",
    "9_img.png":     "399266ca04b55a44677198df608c116550a0cc5b3d7301d79f12af7dc321eb74",
    "11_img.npy":    "4ca021e833f7cfe4207ab236d4c4845704ebef7e45343a30fe502e276d3112fb",
    "11_latent.npy": "b9bb10b015740e3cad536d516458fd64a8839f75f56bcfb9de8260f4bbe7291a",
    "11_img.png":    "426b01cb03199dce140e5031b2b946ecf0c164fdeef01930064b5823e791ca92",
    "13_img.npy":    "d1509b161eba2684f52b4b826a5b9c425632df55f5f3de883b0fa0b4e62862b4",
    "13_latent.npy": "cf3295a79fe921cfc3794a573abf207ccc18645c5e8a18467640f43514d38271",
    "13_img.png":    "8f5cf8e48e04841ee18873d3dbb15c7744b45ce0c8c262560aad0994c028e6ec",
    "15_img.npy":    "269f6c57345c351c8377314202a647dd00b0af19f39476ecb50e74f4713ac434",
    "15_latent.npy": "f275f6d685b64068aa5776dfeb8c1b315e9bbcee55c3c5327958e6dbee4ad5ef",
    "15_img.png":    "f98aee9c6a6560673159dba9370d63aa648d1710085da7c999f7dc8c3502e81a",
    "16_img.npy":    "a530c9fa3fdb580df0fb6bf419a2b453fa4b1665a36ae739139dd06b306c3bcc",
    "16_latent.npy": "dd0dc7456abfb309c11c5342a46e649d8c60f334e6df93c0507bf2b319cbd8c0",
    "16_img.png":    "45b35cab19c35e17adaea3747aed44a8c82ea4f6943ef431219dd1b905d00621",
    "18_img.npy":    "d388a276241a339501c045366226e39243420762946bdcd887806dd6aadcd4e6",
    "18_latent.npy": "b4353da7d95be7a5587ad33e1f04590f29d42be33331c0471bcd19bc5c9f4310",
    "18_img.png":    "d10c30e6265c62591a62a421d9cca391ed9fc759cc116b93f9ab482d13cb7fbc",
    "20_img.npy":    "cf34490ae5303a1157a55bcec35a5ddc3ccad62e04b35e652f956cbbb3f3aa44",
    "20_latent.npy": "c11c2a477d2809e8137c3e21045db07a05a5ac664b9e6e2f1badec942a791308",
    "20_img.png":    "63677b76bbdcc944fd91c83b95f4b158367d959b059cafb762ac26425ce6a0ee",
    # GAN artifacts (replace pickle.load with safetensors + JSON)
    "gan_weights.safetensors": "3897bd712f7db7619c6e32eee8955e2d6714fcb4a3fdbc7f56267c2c69129d19",
    "gan_static_kwargs.json":  "9215b500feeb4bcaa5d388d9f8087a07be3eb065d8321248ce8f4ac5757941d6",
}


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_verified(url, dest_path, expected_hash):
    """Stream url to dest_path, verify SHA-256, atomically move into place.

    Writes to dest_path + ".part" first, hashing as bytes arrive. On match,
    os.replace into dest_path. On mismatch or any error, removes the partial
    file and raises with the offending name + expected/actual hashes.

    Redirects are followed: Hugging Face's `resolve/main/` URLs 302 to a signed
    CDN URL (cas-bridge.xethub.hf.co) as part of normal delivery. The SHA-256
    pin enforces integrity regardless of which host actually serves the bytes,
    so a rogue redirect cannot forge a body that matches the pinned digest.
    """
    tmp_path = dest_path + ".part"
    h = hashlib.sha256()
    try:
        resp = requests.get(url, stream=True, timeout=30, allow_redirects=True, verify=True)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                h.update(chunk)
                f.write(chunk)
        actual = h.hexdigest()
        if actual != expected_hash:
            raise RuntimeError(
                "SHA-256 mismatch for {}: expected {}, got {}".format(
                    os.path.basename(dest_path), expected_hash, actual
                )
            )
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _fetch_all(url_and_dest_pairs):
    """Download a list of (url, dest_path, name) triples with hash verification.

    Each name must appear in _EXPECTED_SHA256 (fail-closed for unknown files).
    Returns the explicit list of dest paths written — no os.walk / substring
    rescan, which previously matched files like "150" when looking for "15".
    """
    written = []
    for url, dest, name in url_and_dest_pairs:
        if name not in _EXPECTED_SHA256:
            raise RuntimeError("No pinned hash for {} — refusing to download".format(name))
        parent = os.path.dirname(dest)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        expected = _EXPECTED_SHA256[name]
        if os.path.isfile(dest) and _sha256_of(dest) == expected:
            print("Skipping {} (local copy matches pinned hash)".format(name))
            written.append(dest)
            continue
        _download_verified(url, dest, expected)
        written.append(dest)
    return written


class dwnld_CEM_MAF_celebA():
    '''
    Class with functions to download
        1. celebA prediction model
        2. celebA attribute functions
        3. celebA data files
        4. GAN artifacts (safetensors weights + JSON sidecar)

    All downloads are over HTTPS with SHA-256 verification against a pinned
    manifest. Files whose bytes don't match their pinned hash are rejected
    and removed.
    '''

    def dwnld_celebA_attributes(self, local_path, attributes):
        '''
        Download celebA attribute functions.

        Args:
            local_path (str): local path to where files are downloaded
                (attribute files land in `<local_path>/attr_model/`).
            attributes (str list): list of attributes to download attribute
                functions for.

        Returns:
            files (str list): list of files that were downloaded.
        '''
        # Only the JSON architecture and H5 weights are runtime-required — the
        # explainers do `model_from_json(...)` + `load_weights(...)`. The
        # historical `.ckpt` files were TF1 training checkpoints and are not
        # loaded by any code path; not mirrored on Hugging Face.
        pairs = []
        for attr in attributes:
            for suffix in ("_model.json", "_weights.h5"):
                name = "simple_" + attr + suffix
                url = _MODEL_BASE_URL + "attr_model/" + name
                dest = os.path.join(local_path, "attr_model", name)
                pairs.append((url, dest, name))
        files = _fetch_all(pairs)
        print("Attribute files downloaded:")
        print(files)
        return files

    def dwnld_celebA_model(self, local_path):
        '''
        Download celebA model.

        Args:
            local_path (str): local path to where the model file is downloaded.

        Returns:
            files (str list): list of files that were downloaded.
        '''
        name = "celebA"
        pairs = [(_MODEL_BASE_URL + name, os.path.join(local_path, name), name)]
        files = _fetch_all(pairs)
        print("celebA model file downloaded:")
        print(files)
        return files

    def dwnld_celebA_data(self, local_path, ids):
        '''
        Download celebA data files.

        The remote layout on Hugging Face is flat (files live at the dataset
        repo root, no `celeba_data/` prefix), while locally they land under
        `local_path` (typically `aix360/data/celeba_data/`). This method
        constructs the URL and destination path independently to keep the
        remote/local asymmetry contained.

        Args:
            local_path (str): local path to where files are downloaded.
            ids (int list): list of ids to download data for. The available
                ids on the dataset repo are [2, 3, 4, 9, 11, 13, 15, 16, 18, 20].

        Returns:
            files (str list): list of files that were downloaded.
        '''
        pairs = []
        for id_ in ids:
            for suffix in ("_img.npy", "_latent.npy", "_img.png"):
                name = str(id_) + suffix
                url = _DATA_BASE_URL + name
                dest = os.path.join(local_path, name)
                pairs.append((url, dest, name))
        files = _fetch_all(pairs)
        print("Image files downloaded:")
        print(files)
        return files

    def dwnld_celebA_gan(self, local_path):
        '''
        Download GAN safetensors weights + JSON sidecar.

        Args:
            local_path (str): destination directory (typically
                `aix360/models/CEM_MAF/gan/`).

        Returns:
            files (str list): list of files that were downloaded.
        '''
        pairs = []
        for name in ("gan_weights.safetensors", "gan_static_kwargs.json"):
            url = _MODEL_BASE_URL + "gan/" + name
            dest = os.path.join(local_path, name)
            pairs.append((url, dest, name))
        files = _fetch_all(pairs)
        print("GAN files downloaded:")
        print(files)
        return files
