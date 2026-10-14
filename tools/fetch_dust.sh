#!/bin/bash
# Bayestar19 (Green et al. 2019) and SFD (Schlegel et al. 1998) from Harvard Dataverse, where dustmaps.fetch expects them
# (with curl: Dataverse refuses dustmaps.fetch's python-requests downloads, 403); each file checked against its MD5
D=$HOME/dustmaps_data
mkdir -p $D/bayestar $D/sfd
get() {  # file id, destination, md5
  [ -f "$2" ] && [ "$(md5sum "$2" 2>/dev/null | cut -d' ' -f1 || md5 -q "$2")" = "$3" ] && { echo "have $2"; return; }
  curl -sS -L --retry 5 -o "$2" "https://dataverse.harvard.edu/api/access/datafile/$1"
  sum=$( (md5sum "$2" 2>/dev/null || md5 -r "$2") | cut -d' ' -f1)
  [ "$sum" = "$3" ] && echo "ok $2" || echo "MD5 MISMATCH $2 ($sum)"
}
get 2902687 $D/sfd/SFD_dust_4096_ngp.fits 6f3a89528bba7d1b7bd55838da86b639
get 2902695 $D/sfd/SFD_dust_4096_sgp.fits 60f7c43d51bccaff82961de3e9f9e789
get 3424724 $D/bayestar/bayestar2019.h5 ab815d2fd3068d1b81a1bd61fb18a722
echo FETCH DONE
