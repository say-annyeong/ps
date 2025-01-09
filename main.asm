default rel
section .data
    input: db "%d %d",0
    output: db "%d",10,0
    a: dd 0
    b: dd 0

section .text
    global main
    extern scanf
    extern printf

main:
    push rbp
    mov rbp, rsp
    sub rsp, 32  ; 스택 정렬

    lea rcx, [rel input]
    lea rdx, [a]
    lea r8, [b]
    call scanf

    mov ecx, [a]
    add ecx, [b]

    lea rcx, [rel output]
    mov edx, ecx
    call printf

    add rsp, 32
    pop rbp
    xor rax, rax
    ret
