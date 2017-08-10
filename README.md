# speech_transf_disco
DiscoGAN + Speech transfer

DiscoGAN implementation is given by https://github.com/carpedm20/DiscoGAN-pytorch

Data : ChiME3

v1 : 
  - A : Clean VS. B : Clean + Bus noise
  - Linear Spectrogram magnitude
  - 257frames x 257fftpoint --> resized to 64x64 (in paper)
  
  
Result 
 - learning progress by saving A, AB, B, BA image for every 500 iters
 - NOTE : this result need to be improved after solving following ISSUES.
  
ISSUES & TODO
1) v1 implementation apply cmvn with mean/var given in 'each patch', not entire patches
   --> do cmvn globally ==> DONE
   
2) image loader does not support non-negative value
   --> find efficient non-image based loader
   --> pickle takes really huge memory for saving lists of numpy array
   
3) Graffin-Lim reconstruction
   --> have code, but not tested
   --> should include inverse cmvn --> ISSUE1&2 need to be addressed
