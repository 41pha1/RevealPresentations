
var SlipSimulation = {

    slipODE: function (x, xi, params) {
        var y = x.e(1);
        var v = x.e(2);

        var k = params.k;
        var m = params.m;
        var b = params.b;
        var g = params.g;

        var spring_acc = k/m*(- y) - b/m*v + xi/m * (v > 0);
        var y_dot_dot = -g + ((- y) > 0) * spring_acc;

        return $V([v, y_dot_dot]);
    },

    residual: function (x, a, params) {
        const x0 = $V([x.e(1), x.e(2)]);
        const xi = x.e(3);
        const T = x.e(4);

        const result = this.simulateSlip(T, 0.002, x0, xi, params);
        const x_T  = result.x[result.x.length - 1];

        return $V([
            x_T.e(1) - x0.e(1),
            x_T.e(2) - x0.e(2),
            x0.e(1) - a,
            x0.e(2)
        ]);
    },

    luDecomp: function (A) {
        const n = A.rows();
        const L = Matrix.I(n);
        const U = A.dup();

        for (var i = 1; i < n; i++) {
            for (var j = i + 1; j <= n; j++) {
                var factor = U.e(j, i) / U.e(i, i);
                L.elements[j-1][i-1] = factor;
                U.elements[j-1][i-1] = 0;

                for (var k = i + 1; k <= n; k++) {
                    U.elements[j-1][k-1] -= factor * U.e(i, k);
                }
            }
        }

        return { L: L, U: U };
    },

    solveLU: function (A, b) {
        const { L, U } = this.luDecomp(A);
        const n = A.rows();
        const y = Vector.Zero(n);
        const x = Vector.Zero(n);

        for (var i = 1; i <= n; i++) {
            y.elements[i-1] = b.e(i);

            for (var j = 1; j < i; j++) {
                y.elements[i-1] -= L.e(i, j) * y.e(j);
            }

            y.elements[i-1] /= L.e(i, i);
        }

        for (var i = n; i > 0; i--) {
            x.elements[i-1] = y.e(i);

            for (var j = i + 1; j <= n; j++) {
                x.elements[i-1] -= U.e(i, j) * x.e(j);
            }

            x.elements[i-1] /= U.e(i, i);
        }

        return x;
    },

    jacobian: function (f, x, fx) {
        const n = x.dimensions();
        const m = fx.dimensions();
        const J = Matrix.Zero(m, n);
        const eps = 1e-8;

        for (var i = 1; i <= n; i++) {
            var x_i = x.dup();
            x_i.elements[i-1] += eps;
            var fx_i = f(x_i);
            var df_i = fx_i.subtract(fx).x(1/eps);
            
            for (var j = 1; j <= m; j++) {
                J.elements[j-1][i-1] = df_i.e(j);
            }
        }

        return J;
    },

    fsolve: function (f, x_init) {
        const tolerance = 1e-10;

        var max_iterations = 15;
        var fx = f(x_init);

        while (fx.modulus() > tolerance && --max_iterations > 0) {
            var J = this.jacobian(f, x_init, fx);
            var dx = this.solveLU(J, fx);
            x_init = x_init.subtract(dx);

            fx = f(x_init);
        }

        console.log("Residual: " + fx.modulus());
        return x_init;
    },

    findPeriodicSolution: function (x0, xi0, T0, a, params) {
        const x_init = $V([x0.e(1), x0.e(2), xi0, T0]);
        const root = this.fsolve((x) => this.residual(x, a, params), x_init);

        return { x: root.e(1), v: root.e(2), xi: root.e(3), T: root.e(4) };
    },

    getSolutionTangent: function (u, params) {

        const residual_prime = (x) => this.residual( $V([x.e(1), x.e(2), x.e(3), x.e(4)]), x.e(5), params );
        const J = this.jacobian(residual_prime, u, residual_prime(u));

        const JtJ = J.transpose().x(J);
        const eigs = math.eigs(JtJ.elements);

        var t = Vector.Zero( J.cols() );
        t.elements = eigs.eigenvectors[0].vector;

        // console.log("Eigenvalue: " + eigs.values[0] + ", Error: " + J.x(t).modulus());
        // console.log("Tangent: x0 = " + t.e(1) + ", v0 = " + t.e(2) + ", xi = " + t.e(3) + ", T = " + t.e(4) + ", a = " + t.e(5));

        // const A = Matrix.Zero( J.rows() + 1, J.cols() );
        // for (var i = 1; i <= J.rows(); i++) {
        //     for (var j = 1; j <= J.cols(); j++) {
        //         A.elements[i-1][j-1] = J.e(i, j);
        //     }
        // }
        // for (var j = 1; j <= J.cols(); j++) {
        //     A.elements[J.rows()][j-1] = t.e(j);
        // }
        // if (A.determinant() < 0) {
        //     t = t.x(-1);
        // }

        if ( t.e(5) > 0 ) {
            t = t.x(-1);
        }

        return t;
    },

    rk4: function(x, t, h, f) {
        var k1 = f(x, t);
        var k2 = f(x.add(k1.x(h/2)), t + h/2);
        var k3 = f(x.add(k2.x(h/2)), t + h/2);
        var k4 = f(x.add(k3.x(h)), t + h);

        return x.add(k1.add(k2.x(2)).add(k3.x(2)).add(k4).x(h/6));
    },

    midpoint: function(x, t, h, f) {
        var k1 = f(x, t);
        var k2 = f(x.add(k1.x(h/2)), t + h/2);

        return x.add(k2.x(h));
    },

    euler: function(x, t, h, f) {
        return x.add(f(x, t).x(h));
    },

    simulateSlip: function (T, h, x0, xi, params, I = "runge-kutta" ) { 
        T = Math.max(T, 0.002);
        var n = Math.floor(T / h) + 1;
        var t_list = Array.from({length: n}, (_, i) => i * T / n);
        var x_list = Array(n);
        var integrator = {"euler": this.euler, "midpoint": this.midpoint, "runge-kutta": this.rk4}[I.toLowerCase()] || this.rk4;

        x_list[0] = x0;

        for (var i = 1; i < n; i++) {
            var dt = t_list[i] - t_list[i-1];
            x_list[i] = integrator(x_list[i-1], t_list[i-1], dt, (x, t) => this.slipODE(x, xi, params));
        }

        return { x: x_list, t: t_list };
    }
}