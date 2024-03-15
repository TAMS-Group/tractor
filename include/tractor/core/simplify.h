// 2020-2024 Philipp Ruppel

#pragma once

namespace tractor {

class Program;

void simplify(Program &program);

void compressZeroConstants(Program &program);
void removeDuplicateConstants(Program &program);
void removeUnusedConstants(Program &program);
void removeUnusedInstructions(Program &program);
void precomputeConstants(Program &program);
void skipMoves(Program &program);
void defragmentMemory(Program &program);

}  // namespace tractor
