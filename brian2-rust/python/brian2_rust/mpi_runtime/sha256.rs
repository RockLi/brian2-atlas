// SHA-256 (FIPS 180-4), used only while loading immutable rank-local input.
// No full input copy and no validation/reopen race: hash exactly consumed bytes.
#[derive(Clone)]
struct Sha256 { state: [u32;8], block: [u8;64], used: usize, length: u64 }
impl Sha256 {
    fn new() -> Self { Self { state: [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19], block: [0;64], used:0, length:0 } }
    fn update(&mut self, mut bytes: &[u8]) {
        self.length=self.length.checked_add(bytes.len() as u64).expect("SHA-256 length overflow");
        while !bytes.is_empty() {
            let n=(64-self.used).min(bytes.len());
            self.block[self.used..self.used+n].copy_from_slice(&bytes[..n]);
            self.used+=n; bytes=&bytes[n..];
            if self.used==64 { self.compress(); self.used=0; }
        }
    }
    fn compress(&mut self) {
        const K:[u32;64]=[
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
        let mut w=[0u32;64];
        for (v,b) in w.iter_mut().zip(self.block.chunks_exact(4)) { *v=u32::from_be_bytes(b.try_into().unwrap()); }
        for i in 16..64 {
            let a=w[i-15]; let b=w[i-2];
            w[i]=w[i-16].wrapping_add(a.rotate_right(7)^a.rotate_right(18)^(a>>3))
                .wrapping_add(w[i-7]).wrapping_add(b.rotate_right(17)^b.rotate_right(19)^(b>>10));
        }
        let [mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut h]=self.state;
        for i in 0..64 {
            let t1=h.wrapping_add(e.rotate_right(6)^e.rotate_right(11)^e.rotate_right(25))
                .wrapping_add((e&f)^(!e&g)).wrapping_add(K[i]).wrapping_add(w[i]);
            let t2=(a.rotate_right(2)^a.rotate_right(13)^a.rotate_right(22))
                .wrapping_add((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d.wrapping_add(t1);d=c;c=b;b=a;a=t1.wrapping_add(t2);
        }
        for (old,new) in self.state.iter_mut().zip([a,b,c,d,e,f,g,h]) { *old=old.wrapping_add(new); }
    }
    fn finish(&self) -> [u8;32] {
        let mut hash=self.clone(); let bits=hash.length.checked_mul(8).expect("SHA-256 bit length overflow");
        hash.update(&[0x80]);
        while hash.used!=56 { hash.update(&[0]); }
        hash.update(&bits.to_be_bytes());
        let mut result=[0u8;32];
        for (bytes,value) in result.chunks_exact_mut(4).zip(hash.state) { bytes.copy_from_slice(&value.to_be_bytes()); }
        result
    }
}
