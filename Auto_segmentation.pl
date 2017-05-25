#!/usr/bin/perl

#
# automatic segmentation wavfile
#

use File::Basename;
use strict;
#require "sox_funcs.pl";

my $wavpath = $ARGV[0];
my $lapse = $ARGV[1];
my $outpath = $ARGV[2];
my $ret;
print "Automatic segmentation script\n";
print"###############################################\n";
if ($#ARGV < 2) {
print"USAGE : auto_segmentation.pl <input_path> <lapse> <out_path>\n";
print "  a list of wave files is loaded from the directory <input_path> (e.g. '/my/path/*')\n";
print "  and will be putted in the folder <out_path> (e.g. '/my/path')\n";
print "  <lapse> is the lenght of each new segment\n";
exit 1;
}

use File::Glob qw(bsd_glob);
print "Getting list of wave files from $wavpath ...\n";

my @wavs = bsd_glob("$wavpath");
#my @wavs= glob("*.wav");
print "    ".($#wavs+1)."files in list.\n";
#print "@wavs\n";

     
  if ($#wavs < 0) {
  print "ERROR: no files found.\n";
  exit 1;}   

	for (my $j=0; $j < $#wavs+1; $j++){  
	print "segmenting file $j\n";
	$ret = autrim($wavs[$j],$lapse,$outpath);
	if ($ret ne "done"){
	print "ERROR!!!! check autrim function\n";
	exit -1;}  
	}
    

sub autrim {
  my $wav_name = shift(@_);
  my $lapse = shift(@_);
  my $out = shift(@_);
  my $ext = "_seg"; 
  my $wav_base_name = basename($wav_name);
  my $durate;
  #$wav_base_name =~ s{.*/}{}; #removes path
  $wav_base_name =~ s{\.[^.]+$}{}; # removes extension 
  $wav_base_name =~ s/\.//g;
  print "$wav_base_name\n";
  

  my $cmd_duration = "soxi -D $wav_name > length_tmp.txt";
#print "$cmd_duration\n";
  my $ret = system($cmd_duration);
#exit 1;
  open LEN, "<", "length_tmp.txt";
  $durate=<LEN>;
  close LEN;
  #unlink "length_tmp.txt";
  my $n_seg = $durate/$lapse;


  if($wav_name =~ m/wav/){
     			for (my $i=0; $i < $n_seg ; $i++)    
				{
     				my $index = $i*$lapse;
				#print "$i\n";
				#print "$index\n";
				my $ext =$ext.$i;
				my $cmd = "sox $wav_name $out$wav_base_name$ext.wav trim $index $lapse";
    				my $ret = system($cmd);	
				#print $cmd."\n";
    				if($ret){
     	 				print "ERROR!!!!!!! $cmd \n";
      					return "error";
        				}
				}
			}
			else {return "done"
			}
    return "done";	      
}
 


print "All done. The result is in: $outpath\n";
