use std::{fs, io::{self, Write}, path::Path};
use serde::{Serialize, Deserialize};
use crate::ivf::{IvfIndex, ListFlat};
use crate::header::IndexHeader;

#[derive(Serialize, Deserialize)]
struct Manifest { header: IndexHeader, fingerprint: u64 }

impl IvfIndex {
    pub fn save_dir(&self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let man = Manifest { header: self.header.clone(), fingerprint: self.fingerprint() };
        fs::write(dir.join("header.json"), serde_json::to_vec_pretty(&man).unwrap())?;

        // centroids
        let mut buf: Vec<u8> = Vec::with_capacity(self.centroids.len() * self.header.dim * 4);
        for c in &self.centroids { for &f in c { buf.extend_from_slice(&f.to_le_bytes()); } }
        fs::write(dir.join("centroids.f32"), buf)?;

        // lists v2
        let mut w = fs::File::create(dir.join("lists.bin"))?;
        let nlist = self.lists.len() as u32; w.write_all(&nlist.to_le_bytes())?;
        for list in &self.lists {
            let cnt = list.ids.len() as u32; w.write_all(&cnt.to_le_bytes())?;
            for &t in &list.tomb { w.write_all(&[t])?; }                 // tomb
            for &id in &list.ids { w.write_all(&id.to_le_bytes())?; }    // ids
            for &f in &list.vecs { w.write_all(&f.to_le_bytes())?; }     // vecs
            let has_tags = if list.tags.as_ref().map_or(false, |t| !t.is_empty()) { 1u8 } else { 0u8 };
            w.write_all(&[has_tags])?;
            if has_tags == 1 { for &mask in list.tags.as_ref().unwrap() { w.write_all(&mask.to_le_bytes())?; } }
        }
        fs::write(dir.join("fingerprint.txt"), format!("{}", self.fingerprint()))?;
        Ok(())
    }

    pub fn load_dir(dir: &Path) -> io::Result<IvfIndex> {
        let man_bytes = fs::read(dir.join("header.json"))?;
        let man: Manifest = serde_json::from_slice(&man_bytes).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        if man.header.version < 2 { return Err(io::Error::new(io::ErrorKind::Other, "unsupported index version")); }

        // centroids
        let cent_bytes = fs::read(dir.join("centroids.f32"))?;
        let mut centroids: Vec<Vec<f32>> = Vec::new();
        let dim = man.header.dim;
        let mut off = 0;
        while off < cent_bytes.len() {
            let mut row = vec![0f32; dim];
            for d in 0..dim { let mut b = [0u8;4]; b.copy_from_slice(&cent_bytes[off..off+4]); off+=4; row[d] = f32::from_le_bytes(b); }
            centroids.push(row);
        }

        // lists v2
        let lb = fs::read(dir.join("lists.bin"))?;
        let mut p = 0usize;
        let mut u32_at = |p:&mut usize| { let mut b=[0u8;4]; b.copy_from_slice(&lb[*p..*p+4]); *p+=4; u32::from_le_bytes(b) as usize };
        let nlist = u32_at(&mut p);
        let mut lists: Vec<ListFlat> = Vec::with_capacity(nlist);
        for _ in 0..nlist {
            let cnt = u32_at(&mut p);
            let mut tomb = Vec::with_capacity(cnt); for _ in 0..cnt { tomb.push(lb[p]); p+=1; }
            let mut ids = Vec::with_capacity(cnt);
            for _ in 0..cnt { let mut b=[0u8;8]; b.copy_from_slice(&lb[p..p+8]); p+=8; ids.push(u64::from_le_bytes(b)); }
            let mut vecs = Vec::with_capacity(cnt*dim);
            for _ in 0..cnt*dim { let mut b=[0u8;4]; b.copy_from_slice(&lb[p..p+4]); p+=4; vecs.push(f32::from_le_bytes(b)); }
            let has_tags = lb[p]; p+=1;
            let tags = if has_tags == 1 {
                let mut t = Vec::with_capacity(cnt);
                for _ in 0..cnt { let mut b=[0u8;8]; b.copy_from_slice(&lb[p..p+8]); p+=8; t.push(u64::from_le_bytes(b)); }
                Some(t)
            } else { None };
            lists.push(ListFlat { ids, vecs, tomb, tags });
        }

        let header = man.header;           // move header out of manifest
        let params = header.params;        // IvfParams is Copy, this is fine
        let idx = IvfIndex { header, params, centroids, lists };

        if idx.fingerprint() != man.fingerprint {
            return Err(io::Error::new(io::ErrorKind::Other, "fingerprint mismatch"));
        }
        Ok(idx)
    }
}
