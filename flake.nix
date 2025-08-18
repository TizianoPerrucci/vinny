{
  description = "A very basic flake";

    inputs = {
      flake-utils.url = "github:numtide/flake-utils";
      nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    };

    outputs = { self, nixpkgs, flake-utils }:
      flake-utils.lib.eachDefaultSystem (system:
        let
          pkgs = import nixpkgs { inherit system; };

          poetryDeps = pkgs: (with pkgs; [
            poetry
            just
          ]);
        in
        {
          devShells.default = pkgs.mkShell {
            buildInputs = poetryDeps (pkgs);
            shellHook = ''
              mkdir -p sys-lib
              ln -sf /usr/lib/x86_64-linux-gnu/libcuda* sys-lib

              ln -sf /usr/lib/x86_64-linux-gnu/libGL* sys-lib
              ln -sf /usr/lib/x86_64-linux-gnu/libgthread* sys-lib
              ln -sf /usr/lib/x86_64-linux-gnu/libglib* sys-lib
              ln -sf /usr/lib/x86_64-linux-gnu/libpcre* sys-lib
              ln -sf /usr/lib/x86_64-linux-gnu/libX* sys-lib
              ln -sf /usr/lib/x86_64-linux-gnu/libxcb* sys-lib
              ln -sf /usr/lib/x86_64-linux-gnu/libbsd* sys-lib
              ln -sf /usr/lib/x86_64-linux-gnu/libmd* sys-lib

              export LD_LIBRARY_PATH="${with pkgs; lib.makeLibraryPath [stdenv.cc.cc.lib zlib]}:./sys-lib";
            '';
          };
        }
      );

}
