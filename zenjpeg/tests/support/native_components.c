/* Test-only libjpeg-turbo oracle for component layouts not exposed by cjpeg's
 * PNM input. Binary samples use big-endian u16 so results are host-independent.
 * Usage: encode precision channels lossless predictor point sampling progressive
 *        decode raw|rgb|gray|cmyk
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jpeglib.h"

int main(int argc, char **argv) {
  struct jpeg_error_mgr err;
  if(argc==2 && strcmp(argv[1],"version")==0) { printf("%d\n", LIBJPEG_TURBO_VERSION_NUMBER); return 0; }
  if ((argc == 9 || argc == 11) && strcmp(argv[1], "encode") == 0) {
    int bits=atoi(argv[2]), n=atoi(argv[3]), lossless=atoi(argv[4]);
    if(n<1 || n>10 || bits<2 || bits>16) return 2;
    struct jpeg_compress_struct c;
    c.err=jpeg_std_error(&err); jpeg_create_compress(&c);
    jpeg_stdio_dest(&c, stdout);
    c.image_width=argc==11 ? atoi(argv[9]) : 35;
    c.image_height=argc==11 ? atoi(argv[10]) : 33; c.input_components=n;
    if(c.image_width<1 || c.image_width>35 || c.image_height<1 || c.image_height>33) return 2;
    c.in_color_space=n==4 ? JCS_CMYK : JCS_UNKNOWN;
    jpeg_set_defaults(&c); c.data_precision=bits;
    jpeg_set_quality(&c,73,FALSE);
    if (lossless) jpeg_enable_lossless(&c,atoi(argv[5]),atoi(argv[6]));
    else if (n==4 && atoi(argv[7])>1) jpeg_set_colorspace(&c,JCS_YCCK);
    if (!lossless && atoi(argv[7])>1) {
      c.comp_info[0].h_samp_factor=atoi(argv[7]);
      c.comp_info[0].v_samp_factor=1;
      if(n==4) { c.comp_info[3].h_samp_factor=atoi(argv[7]); c.comp_info[3].v_samp_factor=1; }
    }
    int mode=atoi(argv[8]);
    jpeg_scan_info scans[10];
    if (n>4) {
      for (int t=0;t<4;t++) {
        unsigned int quant[64];
        for(int k=0;k<64;k++) quant[k]=3+t*7+k%5;
        jpeg_add_quant_table(&c,t,quant,100,FALSE);
      }
      for(int k=0;k<n;k++) {
        c.comp_info[k].quant_tbl_no=lossless ? 0 : k%4;
        c.comp_info[k].dc_tbl_no=mode>=2 ? k+(mode==3 ? 6 : 0) : k%4;
        c.comp_info[k].ac_tbl_no=mode>=2 ? k+(mode==3 ? 6 : 0) : k%4;
      }
      memset(scans,0,sizeof(scans));
      int scan=0;
      for(int start=0;start<n;start+=4) {
        int count=n-start<4 ? n-start : 4;
        scans[scan].comps_in_scan=count;
        for(int j=0;j<count;j++) scans[scan].component_index[j]=start+j;
        scans[scan].Ss=lossless ? atoi(argv[5]) : 0;
        scans[scan].Se=lossless ? 0 : 63;
        scans[scan].Al=lossless ? atoi(argv[6]) : 0;
        scan++;
      }
      c.scan_info=scans; c.num_scans=scan;
    }
    if (n>4 && atoi(argv[7])>=5) {
      c.comp_info[0].h_samp_factor=atoi(argv[7])==5 ? 1 : 2;
      c.comp_info[0].v_samp_factor=2;
    }
    if (mode%2) jpeg_simple_progression(&c);
    c.arith_code=mode>=2;
    c.restart_in_rows=2;
    c.optimize_coding=!c.arith_code;
    jpeg_start_compress(&c,TRUE);
    unsigned char row8[35*10]; short row12[35*10]; unsigned short row16[35*10];
    while(c.next_scanline<c.image_height) {
      unsigned y=c.next_scanline, mask=(1u<<bits)-1;
      for(unsigned x=0;x<c.image_width;x++) for(int k=0;k<n;k++) {
        unsigned v=(x*197+y*331+k*613+x*y*29)&mask;
        row8[x*n+k]=v; row12[x*n+k]=v; row16[x*n+k]=v;
      }
      if(bits<=8) { JSAMPROW r=row8; jpeg_write_scanlines(&c,&r,1); }
      else if(bits<=12) { J12SAMPROW r=row12; jpeg12_write_scanlines(&c,&r,1); }
      else { J16SAMPROW r=row16; jpeg16_write_scanlines(&c,&r,1); }
    }
    jpeg_finish_compress(&c); jpeg_destroy_compress(&c); return 0;
  }
  if(argc==3 && strcmp(argv[1],"decode")==0) {
    struct jpeg_decompress_struct c;
    c.err=jpeg_std_error(&err); jpeg_create_decompress(&c);
    jpeg_stdio_src(&c,stdin); jpeg_read_header(&c,TRUE);
    c.out_color_space=strcmp(argv[2],"cmyk")==0 ? JCS_CMYK :
                      strcmp(argv[2],"rgb")==0 ? JCS_RGB :
                      strcmp(argv[2],"gray")==0 ? JCS_GRAYSCALE : c.jpeg_color_space;
    jpeg_start_decompress(&c);
    unsigned count=c.output_width*c.output_components;
    unsigned char *row8=malloc(count); short *row12=malloc(count*2); unsigned short *row16=malloc(count*2);
    if(!row8 || !row12 || !row16) return 2;
    while(c.output_scanline<c.output_height) {
      if(c.data_precision<=8) { JSAMPROW r=row8; jpeg_read_scanlines(&c,&r,1); }
      else if(c.data_precision<=12) { J12SAMPROW r=row12; jpeg12_read_scanlines(&c,&r,1); }
      else { J16SAMPROW r=row16; jpeg16_read_scanlines(&c,&r,1); }
      for(unsigned i=0;i<count;i++) {
        unsigned v=c.data_precision<=8 ? row8[i] : c.data_precision<=12 ? (unsigned short)row12[i] : row16[i];
        putchar(v>>8); putchar(v&255);
      }
    }
    jpeg_finish_decompress(&c); jpeg_destroy_decompress(&c);
    free(row8); free(row12); free(row16); return 0;
  }
  fprintf(stderr,"invalid oracle arguments\n"); return 2;
}
