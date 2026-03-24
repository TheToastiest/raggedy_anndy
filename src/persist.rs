use std::{fs, io::{self, Write}, path::Path};
use serde::{Serialize, Deserialize};
use crate::ivf::{IvfIndex, ListFlat};
use crate::header::IndexHeader;

#[derive(Serialize, Deserialize)]
struct Manifest { header: IndexHeader, fingerprint: u64 }

impl IvfIndex {
    pub fn save_dir(&self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;

        // 1. Header + Fingerprint
        let man = Manifest { header: self.header.clone(), fingerprint: self.fingerprint() };
        fs::write(dir.join("header.json"), serde_json::to_vec_pretty(&man).unwrap())?;

        // 2. Centroids (Buffered)
        let cent_file = fs::File::create(dir.join("centroids.f32"))?;
        let mut cw = io::BufWriter::new(cent_file);
        for c in &self.centroids {
            for &f in c { cw.write_all(&f.to_le_bytes())?; }
        }
        cw.flush()?;

        // 3. Lists v2 (Buffered)
        let list_file = fs::File::create(dir.join("lists.bin"))?;
        let mut w = io::BufWriter::new(list_file);

        let nlist = self.lists.len() as u32;
        w.write_all(&nlist.to_le_bytes())?;

        for list in &self.lists {
            let cnt = list.ids.len() as u32;
            w.write_all(&cnt.to_le_bytes())?;

            // Write tombstone bitmask at once
            w.write_all(&list.tomb)?;

            for &id in &list.ids { w.write_all(&id.to_le_bytes())?; }
            for &f in &list.vecs { w.write_all(&f.to_le_bytes())?; }

            let has_tags = list.tags.as_ref().map_or(false, |t| !t.is_empty());
            w.write_all(&[has_tags as u8])?;
            if has_tags {
                for &mask in list.tags.as_ref().unwrap() {
                    w.write_all(&mask.to_le_bytes())?;
                }
            }
        }
        w.flush()?;

        fs::write(dir.join("fingerprint.txt"), format!("{}", self.fingerprint()))?;
        Ok(())
    }

    pub fn load_dir(dir: &Path) -> io::Result<IvfIndex> {
        let man_bytes = fs::read(dir.join("header.json"))?;
        let man: Manifest = serde_json::from_slice(&man_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        if man.header.version < 2 {
            return Err(io::Error::new(io::ErrorKind::Other, "unsupported index version"));
        }

        // 1. Load Centroids
        let cent_bytes = fs::read(dir.join("centroids.f32"))?;
        let mut centroids = Vec::new();
        let dim = man.header.dim;
        let mut chunks = cent_bytes.chunks_exact(4);

        let mut current_row = Vec::with_capacity(dim);
        for chunk in &mut chunks {
            let mut b = [0u8; 4];
            b.copy_from_slice(chunk);
            current_row.push(f32::from_le_bytes(b));

            if current_row.len() == dim {
                centroids.push(current_row);
                current_row = Vec::with_capacity(dim);
            }
        }

        // 2. Load Lists v2
        let lb = fs::read(dir.join("lists.bin"))?;
        let mut p = 0usize;

        let mut read_bytes = |size: usize| -> io::Result<&[u8]> {
            if p + size > lb.len() {
                Err(io::Error::new(io::ErrorKind::UnexpectedEof, "corrupted lists.bin file"))
            } else {
                let slice = &lb[p..p+size];
                p += size;
                Ok(slice)
            }
        };

        let nlist_bytes = read_bytes(4)?;
        let nlist = u32::from_le_bytes(nlist_bytes.try_into().unwrap()) as usize;
        let mut lists = Vec::with_capacity(nlist);

        for _ in 0..nlist {
            let cnt_bytes = read_bytes(4)?;
            let cnt = u32::from_le_bytes(cnt_bytes.try_into().unwrap()) as usize;

            let tomb = read_bytes(cnt)?.to_vec();

            let mut ids = Vec::with_capacity(cnt);
            for _ in 0..cnt {
                let b = read_bytes(8)?;
                ids.push(u64::from_le_bytes(b.try_into().unwrap()));
            }

            let mut vecs = Vec::with_capacity(cnt * dim);
            for _ in 0..cnt * dim {
                let b = read_bytes(4)?;
                vecs.push(f32::from_le_bytes(b.try_into().unwrap()));
            }

            let has_tags_byte = read_bytes(1)?;
            let has_tags = has_tags_byte[0] == 1;

            let tags = if has_tags {
                let mut t = Vec::with_capacity(cnt);
                for _ in 0..cnt {
                    let b = read_bytes(8)?;
                    t.push(u64::from_le_bytes(b.try_into().unwrap()));
                }
                Some(t)
            } else {
                None
            };

            lists.push(ListFlat { ids, vecs, tomb, tags });
        }

        // FIXED: Extract params first to avoid "Moved Value" error
        let params = man.header.params;
        let header = man.header;
        let fingerprint = man.fingerprint;

        let idx = IvfIndex { header, params, centroids, lists };

        if idx.fingerprint() != fingerprint {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "fingerprint mismatch"));
        }

        Ok(idx)
    }
}