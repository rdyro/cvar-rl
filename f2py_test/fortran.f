      subroutine sample_from_cp(cp, n, r, m, idx)
        integer n
        real*8 cp(n)
        integer m
        real*8 r(m)
        integer*8 idx(m)
cf2py   intent(in) cp
cf2py   intent(in) r
cf2py   intent(out) idx
cf2py   integer, intent(hide), depend(cp) :: n=size(cp)
cf2py   integer, intent(hide), depend(r) :: m=size(r)

        real*8 ri
        integer*8 j

        do i = 1,m
          ri = r(i)
          j = 1
          do while (ri .gt. cp(j) .and. j .lt. n)
            j = j + 1
          enddo
          idx(i) = j - 1
        enddo
        return
      end

      subroutine find_first(fun, x, n, idx)
        logical fun
        external fun

        integer n, idx
        real*8 x(n)
        logical id

cf2py   intent(in) x
cf2py   integer, intent(hide), depend(x) :: n=size(x)
cf2py   intent(out) idx
cf2py   real*8 y
cf2py   id = fun(y)

        do i = 1,n
          id = fun(x(i))
          write (*,*) id, x(i)
          if (id) then
            idx = i
            return
          endif
        enddo
        idx = -1
        return
      end
