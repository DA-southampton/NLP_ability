#!/usr/bin/env perl

# Note: retrieved from https://github.com/apache/incubator-joshua/blob/master/scripts/preparation/detokenize.pl

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

use warnings;
use strict;

# Sample De-Tokenizer
# written by Josh Schroeder, based on code by Philipp Koehn
# modified later by ByungGyu Ahn, bahn@cs.jhu.edu, Luke Orland

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

my $language = "en";
my $QUIET = 1;
my $HELP = 0;

while (@ARGV) {
  $_ = shift;
  /^-l$/ && ($language = shift, next);
  /^-v$/ && ($QUIET = 0, next);
  /^-h$/ && ($HELP = 1, next);
}

if ($HELP) {
  print "Usage ./detokenizer.perl (-l [en|de|...]) < tokenizedfile > detokenizedfile\n";
  exit;
}
if (!$QUIET) {
  print STDERR "Detokenizer Version 1.1\n";
  print STDERR "Language: $language\n";
}

while(<STDIN>) {
  if (/^<.+>$/ || /^\s*$/) {
    #don't try to detokenize XML/HTML tag lines
    print $_;
  }
  else {
    print &detokenize($_);
  }
}

sub detokenize {
  my($text) = @_;
  chomp($text);
  $text = " $text ";

  #  convert curly quotes to ASCII e.g. ‘“”’
  $text =~ s/\x{2018}/'/gs;
  $text =~ s/\x{2019}/'/gs;
  $text =~ s/\x{201c}/"/gs;
  $text =~ s/\x{201d}/"/gs;
  $text =~ s/\x{e2}\x{80}\x{98}/'/gs;
  $text =~ s/\x{e2}\x{80}\x{99}/'/gs;
  $text =~ s/\x{e2}\x{80}\x{9c}/"/gs;
  $text =~ s/\x{e2}\x{80}\x{9d}/"/gs;

  $text =~ s/ '\s+' / " /g;
  $text =~ s/ ` / ' /g;
  $text =~ s/ ' / ' /g;
  $text =~ s/ `` / " /g;
  $text =~ s/ '' / " /g;

  # replace the pipe character, which is
  # a special reserved character in Moses
  $text =~ s/ -PIPE- / \| /g;

  $text =~ s/ -LRB- / \( /g;
  $text =~ s/ -RRB- / \) /g;
  $text =~ s/ -LSB- / \[ /g;
  $text =~ s/ -RSB- / \] /g;
  $text =~ s/ -LCB- / \{ /g;
  $text =~ s/ -RCB- / \} /g;
  $text =~ s/ -lrb- / \( /g;
  $text =~ s/ -rrb- / \) /g;
  $text =~ s/ -lsb- / \[ /g;
  $text =~ s/ -rsb- / \] /g;
  $text =~ s/ -lcb- / \{ /g;
  $text =~ s/ -rcb- / \} /g;

  $text =~ s/ 'll /'ll /g;
  $text =~ s/ 're /'re /g;
  $text =~ s/ 've /'ve /g;
  $text =~ s/ n't /n't /g;
  $text =~ s/ 'LL /'LL /g;
  $text =~ s/ 'RE /'RE /g;
  $text =~ s/ 'VE /'VE /g;
  $text =~ s/ N'T /N'T /g;
  $text =~ s/ can not / cannot /g;
  $text =~ s/ Can not / Cannot /g;

  # just in case the contraction was not properly treated
  $text =~ s/ ' ll /'ll /g;
  $text =~ s/ ' re /'re /g;
  $text =~ s/ ' ve /'ve /g;
  $text =~ s/n ' t /n't /g;
  $text =~ s/ ' LL /'LL /g;
  $text =~ s/ ' RE /'RE /g;
  $text =~ s/ ' VE /'VE /g;
  $text =~ s/N ' T /N'T /g;

  my $word;
  my $i;
  my @words = split(/ /,$text);
  $text = "";
  my %quoteCount =  ("\'"=>0,"\""=>0);
  my $prependSpace = " ";
  for ($i=0;$i<(scalar(@words));$i++) {
    if ($words[$i] =~ /^[\p{IsSc}]+$/) {
      #perform shift on currency
      if (($i<(scalar(@words)-1)) && ($words[$i+1] =~ /^[0-9]/)) {
        $text = $text.$prependSpace.$words[$i];
        $prependSpace = "";
      } else {
        $text=$text.$words[$i];
        $prependSpace = " ";
      }
    } elsif ($words[$i] =~ /^[\(\[\{\¿\¡]+$/) {
      #perform right shift on random punctuation items
      $text = $text.$prependSpace.$words[$i];
      $prependSpace = "";
    } elsif ($words[$i] =~ /^[\,\.\?\!\:\;\\\%\}\]\)]+$/){
      #perform left shift on punctuation items
      $text=$text.$words[$i];
      $prependSpace = " ";
    } elsif (($language eq "en") && ($i>0) && ($words[$i] =~ /^[\'][\p{IsAlpha}]/) && ($words[$i-1] =~ /[\p{IsAlnum}]$/)) {
      #left-shift the contraction for English
      $text=$text.$words[$i];
      $prependSpace = " ";
    } elsif (($language eq "en") && ($i>0) && ($i<(scalar(@words)-1)) && ($words[$i] eq "&") && ($words[$i-1] =~ /^[A-Z]$/) && ($words[$i+1] =~ /^[A-Z]$/)) {
      #some contraction with an ampersand e.g. "R&D"
      $text .= $words[$i];
      $prependSpace = "";
    }  elsif (($language eq "fr") && ($i<(scalar(@words)-1)) && ($words[$i] =~ /[\p{IsAlpha}][\']$/) && ($words[$i+1] =~ /^[\p{IsAlpha}]/)) {
      #right-shift the contraction for French
      $text = $text.$prependSpace.$words[$i];
      $prependSpace = "";
    } elsif ($words[$i] =~ /^[\'\"]+$/) {
      #combine punctuation smartly
      if (($quoteCount{$words[$i]} % 2) eq 0) {
        if(($language eq "en") && ($words[$i] eq "'") && ($i > 0) && ($words[$i-1] =~ /[s]$/)) {
          #single quote for posesssives ending in s... "The Jones' house"
          #left shift
          $text=$text.$words[$i];
          $prependSpace = " ";
        } elsif (($language eq "en") && ($words[$i] eq "'") && ($i < (scalar(@words)-1)) && ($words[$i+1] eq "s")) {
          #single quote for possessive construction. "John's"
          $text .= $words[$i];
          $prependSpace = "";
        } elsif (($quoteCount{$words[$i]} == 0) &&
          ($language eq "en") && ($words[$i] eq '"') && ($i>1) && ($words[$i-1] =~ /^[,.]$/) && ($words[$i-2] ne "said")) {
          #emergency case in which the opening quote is missing
          #ending double quote for direct quotes. e.g. Blah," he said. but not like he said, "Blah.
          $text .= $words[$i];
          $prependSpace = " ";
        } elsif (($language eq "en") && ($words[$i] eq '"') && ($i < (scalar(@words)-1)) && ($words[$i+1] =~ /^[,.]$/)) {
          $text .= $words[$i];
          $prependSpace = " ";
        } else {
          #right shift
          $text = $text.$prependSpace.$words[$i];
          $prependSpace = "";
          $quoteCount{$words[$i]} = $quoteCount{$words[$i]} + 1;

        }
      } else {
        #left shift
        $text=$text.$words[$i];
        $prependSpace = " ";
        $quoteCount{$words[$i]} = $quoteCount{$words[$i]} + 1;

      }

    } else {
      $text=$text.$prependSpace.$words[$i];
      $prependSpace = " ";
    }
  }

  #clean continuing spaces
  $text =~ s/ +/ /g;

  #delete spaces around double angle brackets «»
  # Uh-oh. not a good idea. it is not consistent.
  $text =~ s/(\x{c2}\x{ab}|\x{ab}) /$1/g;
  $text =~ s/ (\x{c2}\x{bb}|\x{bb})/$1/g;

  # delete spaces around all other special characters
  # Uh-oh. not a good idea. "Men&Women"
  #$text =~ s/ ([^\p{IsAlnum}\s\.\'\`\,\-\"\|]) /$1/g;
  $text =~ s/ \/ /\//g;

  # clean up spaces at head and tail of each line as well as any double-spacing
  $text =~ s/\n /\n/g;
  $text =~ s/ \n/\n/g;
  $text =~ s/^ //g;
  $text =~ s/ $//g;

  #add trailing break
  $text .= "\n" unless $text =~ /\n$/;

  return $text;
}
