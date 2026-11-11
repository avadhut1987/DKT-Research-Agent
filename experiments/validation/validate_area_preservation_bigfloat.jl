using LinearAlgebra
using StaticArrays

setprecision(BigFloat, 256) do

    function dkt_map(R, p, k, kp)
        X, Y, Z = R

        cp, sp = cos(p), sin(p)
        X3 = X*cp + Z*sp
        Y3 = Y
        Z3 = Z*cp - X*sp

        ckz, skz = cos(k*Z3), sin(k*Z3)
        X2 = X3*ckz - Y3*skz
        Y2 = Y3*ckz + X3*skz
        Z2 = Z3

        ckp, skp = cos(kp*X2), sin(kp*X2)
        X1 = X2
        Y1 = Y2*ckp - Z2*skp
        Z1 = Z2*ckp + Y2*skp

        return SVector{3,BigFloat}(X1, Y1, Z1)
    end

    function dkt_jacobian(R, p, k, kp)
        X, Y, Z = R

        cp, sp = cos(p), sin(p)
        X3 = X*cp + Z*sp
        Y3 = Y
        Z3 = Z*cp - X*sp

        Jp = [
            cp  zero(cp) sp
            zero(cp) one(cp) zero(cp)
            -sp zero(cp) cp
        ]

        ckz, skz = cos(k*Z3), sin(k*Z3)
        X2 = X3*ckz - Y3*skz
        Y2 = Y3*ckz + X3*skz
        Z2 = Z3

        Jz = [
            ckz -skz -k*(X3*skz + Y3*ckz)
            skz ckz k*(X3*ckz - Y3*skz)
            zero(k) zero(k) one(k)
        ]

        ckp, skp = cos(kp*X2), sin(kp*X2)

        Jx = [
            one(kp) zero(kp) zero(kp)
            -kp*(Y2*skp + Z2*ckp) ckp -skp
            kp*(Y2*ckp - Z2*skp) skp ckp
        ]

        return Jx * Jz * Jp
    end

    function tangent_basis(R)
        reference = abs(R[3]) < big"0.9" ?
            SVector(zero(R[1]), zero(R[1]), one(R[1])) :
            SVector(one(R[1]), zero(R[1]), zero(R[1]))

        e1 = normalize(cross(R, reference))
        e2 = normalize(cross(R, e1))

        return hcat(e1, e2)
    end

    p = acos(big(-1))/2

    parameter_sets = [
        ("0", "1"),
        ("100", "1"),
        ("1000", "1"),
        ("10000", "1"),
        ("100000", "1"),
    ]

    initial_conditions = [
        ("0.75", "-1.0"),
        ("1.0", "-0.75"),
        ("1.1", "0.3"),
        ("0.7853981633974483", "1.0471975511965976"),
        ("1.4", "-2.0"),
    ]

    println("256-BIT DKT AREA-PRESERVATION VALIDATION")
    println("="^65)

    for (kt_string, kr_string) in parameter_sets
        kt = parse(BigFloat, kt_string)
        kr = parse(BigFloat, kr_string)

        k = kr + kt
        kp = kr - kt

        max_det_error = zero(BigFloat)
        max_tangent_error = zero(BigFloat)

        for (theta_string, phi_string) in initial_conditions
            theta = parse(BigFloat, theta_string)
            phi = parse(BigFloat, phi_string)

            R = SVector(
                sin(theta)*cos(phi),
                sin(theta)*sin(phi),
                cos(theta)
            )

            J = dkt_jacobian(R, p, k, kp)
            Rnext = dkt_map(R, p, k, kp)

            B = tangent_basis(R)
            Bnext = tangent_basis(Rnext)

            A = transpose(Bnext)*J*B

            det_error = abs(abs(det(A)) - one(BigFloat))
            tangent_error = norm(transpose(Rnext)*J*B)

            max_det_error = max(max_det_error, det_error)
            max_tangent_error = max(
                max_tangent_error, tangent_error
            )
        end

        println()
        println("kr = ", kr, ", ktheta = ", kt)
        println("Maximum determinant error = ",
                Float64(max_det_error))
        println("Maximum tangent leakage   = ",
                Float64(max_tangent_error))
    end

    println()
    println("256-bit validation completed.")
end
